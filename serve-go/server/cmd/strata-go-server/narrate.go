package main

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
	"time"
)

const narrationReadSize = 32 << 10
const narrationMaxLineSize = 64 << 10

type narrationEvent struct {
	kind string
	data string
}

type startNarrator struct {
	start, last time.Time
	said        map[string]bool
	now         func() time.Time
	write       func(narrationEvent)
}

type narrationLines struct {
	pending []byte
	drop    bool
}

func (l *narrationLines) feed(chunk []byte, line func(string), oversized func()) {
	for len(chunk) > 0 {
		i := bytes.IndexByte(chunk, '\n')
		part := chunk
		if i >= 0 {
			part = chunk[:i]
		}
		if !l.drop {
			if len(l.pending)+len(part) > narrationMaxLineSize {
				l.pending = l.pending[:0]
				l.drop = true
				oversized()
			} else {
				l.pending = append(l.pending, part...)
			}
		}
		if i < 0 {
			return
		}
		if !l.drop {
			line(strings.TrimSuffix(string(l.pending), "\r"))
		}
		l.pending = l.pending[:0]
		l.drop = false
		chunk = chunk[i+1:]
	}
}

func newStartNarrator(now func() time.Time, write func(narrationEvent)) *startNarrator {
	t := now()
	return &startNarrator{start: t, last: t, said: make(map[string]bool), now: now, write: write}
}

func (n *startNarrator) say(kind, data string) {
	if !n.said[kind] {
		n.said[kind] = true
		n.last = n.now()
		n.write(narrationEvent{kind, data})
	}
}

func (n *startNarrator) line(line string) {
	switch {
	case strings.Contains(line, "PLE on"), strings.Contains(line, "expert arena:"), strings.Contains(line, "experts via mmap"):
		n.say("arena", line)
	case strings.Contains(line, " loaded ") && strings.Contains(line, "GiB at"):
		n.say("loaded", strings.TrimSpace(strings.SplitN(line, " loaded ", 2)[1]))
	case strings.Contains(line, "expert cache ") && strings.Contains(line, " slots, ") && !strings.Contains(line, "auto"):
		n.say("cache", strings.TrimSpace(strings.SplitN(strings.SplitN(line, "expert cache ", 2)[1], ";", 2)[0]))
	case strings.Contains(line, "session is up"):
		n.say("up", line)
	}
}

func (n *startNarrator) heartbeat() {
	now := n.now()
	if now.Sub(n.last) >= 20*time.Second {
		n.last = now
		n.write(narrationEvent{"heartbeat", fmt.Sprintf("%.0f", now.Sub(n.start).Seconds())})
	}
}

func narrateStart(path string, offset int64, done <-chan struct{}, joined chan<- struct{}, interval time.Duration, now func() time.Time, write func(narrationEvent)) {
	defer close(joined)
	n := newStartNarrator(now, write)
	n.write(narrationEvent{kind: "weights"})
	var f *os.File
	if path != "" {
		var err error
		f, err = os.Open(path)
		if err == nil {
			var st os.FileInfo
			st, err = f.Stat()
			if err == nil && !st.Mode().IsRegular() {
				err = fmt.Errorf("engine log is not a regular file")
			}
			if err == nil {
				_, err = f.Seek(offset, io.SeekStart)
			}
			if err != nil {
				n.write(narrationEvent{"log_error", err.Error()})
				if closeErr := f.Close(); closeErr != nil {
					n.write(narrationEvent{"log_error", closeErr.Error()})
				}
				f = nil
			}
		} else {
			n.write(narrationEvent{"log_error", err.Error()})
		}
	}
	defer func() {
		if f != nil {
			if err := f.Close(); err != nil {
				n.write(narrationEvent{"log_error", err.Error()})
			}
		}
	}()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	pos := offset
	lines := narrationLines{pending: make([]byte, 0, narrationMaxLineSize)}
	buf := make([]byte, narrationReadSize)
	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			if f != nil {
				count, err := f.ReadAt(buf, pos)
				pos += int64(count)
				lines.feed(buf[:count], n.line, func() {
					n.write(narrationEvent{"log_error", "engine log line exceeds 64 KiB; discarding until newline"})
				})
				if err != nil && err != io.EOF {
					n.write(narrationEvent{"log_error", err.Error()})
					if closeErr := f.Close(); closeErr != nil {
						n.write(narrationEvent{"log_error", closeErr.Error()})
					}
					f = nil
				}
			}
			n.heartbeat()
		}
	}
}
