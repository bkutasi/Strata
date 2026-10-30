package server

import (
	"math"
	"sort"
	"strings"

	"strata/serve-go/modeltokenizer"
	"strata/streamtext"
)

type logprobAlternative struct {
	ID      int
	Logprob float64
}

type logprobInput struct {
	ID           int
	Logprob      float64
	Alternatives []logprobAlternative
}

type logprobTopEntry struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes"`
}

type logprobEntry struct {
	Token       string            `json:"token"`
	Logprob     float64           `json:"logprob"`
	Bytes       []int             `json:"bytes"`
	TopLogprobs []logprobTopEntry `json:"top_logprobs"`
}

type logprobCollector struct {
	tokenizer *modeltokenizer.Tokenizer
	stopIDs   map[int]bool
	allowed   map[int]bool
	decoder   streamtext.Decoder
	pending   []logprobEntry
}

func newLogprobCollector(tokenizer *modeltokenizer.Tokenizer, stopIDs map[int]bool, allowed []int) *logprobCollector {
	c := &logprobCollector{tokenizer: tokenizer, stopIDs: stopIDs}
	if len(allowed) > 0 {
		c.allowed = make(map[int]bool, len(allowed))
		for _, id := range allowed {
			c.allowed[id] = true
		}
	}
	return c
}

func logprobBytes(raw []byte) []int {
	out := make([]int, len(raw))
	for i, b := range raw {
		out[i] = int(b)
	}
	return out
}

// Decode an alternative independently, including a replacement for an incomplete tail.
func logprobText(raw []byte) string {
	var decoder streamtext.Decoder
	text := decoder.Push(raw)
	if decoder.Pending() {
		text += "\ufffd"
	}
	return text
}

func (c *logprobCollector) Add(in logprobInput) error {
	if c.stopIDs[in.ID] || math.IsNaN(in.Logprob) || math.IsInf(in.Logprob, 0) {
		return nil
	}
	for _, alt := range in.Alternatives {
		if math.IsNaN(alt.Logprob) || math.IsInf(alt.Logprob, 0) {
			return nil
		}
	}
	raw, err := c.tokenizer.TokenBytes(in.ID)
	if err != nil {
		return err
	}
	top := make([]logprobTopEntry, 0, len(in.Alternatives)+1)
	own := c.allowed != nil && c.allowed[in.ID]
	if own {
		top = append(top, logprobTopEntry{logprobText(raw), in.Logprob, logprobBytes(raw)})
	}
	for _, alt := range in.Alternatives {
		if c.allowed != nil && !c.allowed[alt.ID] {
			continue
		}
		b, err := c.tokenizer.TokenBytes(alt.ID)
		if err != nil {
			return err
		}
		top = append(top, logprobTopEntry{logprobText(b), alt.Logprob, logprobBytes(b)})
	}
	lp := in.Logprob
	if c.allowed != nil && len(top) > 0 {
		mx := top[0].Logprob
		for _, alt := range top[1:] {
			mx = math.Max(mx, alt.Logprob)
		}
		sum := 0.0
		for _, alt := range top {
			sum += math.Exp(alt.Logprob - mx)
		}
		lse := mx + math.Log(sum)
		for i := range top {
			top[i].Logprob -= lse
		}
		if own {
			lp = top[0].Logprob
		}
		sort.SliceStable(top, func(i, j int) bool { return top[i].Logprob > top[j].Logprob })
	}
	c.pending = append(c.pending, logprobEntry{c.decoder.Push(raw), lp, logprobBytes(raw), top})
	return nil
}

func (c *logprobCollector) Spell(text string) []logprobEntry {
	var joined strings.Builder
	at := make([]int, 1, len(c.pending)+1)
	for _, entry := range c.pending {
		joined.WriteString(entry.Token)
		at = append(at, joined.Len())
	}
	s := joined.String()
	for pos := 0; text != "" && pos <= len(s); {
		i := strings.Index(s[pos:], text)
		if i < 0 {
			break
		}
		i += pos
		for a := 0; a < len(at)-1; a++ {
			if at[a] != i {
				continue
			}
			for b := a + 1; b < len(at); b++ {
				if at[b] == i+len(text) {
					out := c.pending[a:b]
					c.pending = c.pending[b:]
					return out
				}
			}
			break
		}
		pos = i + 1
	}
	out := c.pending
	c.pending = nil
	return out
}
