package server

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"strata/requestnorm"
	"strata/server/visioninput"
)

func removeImageFile(path string) {
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		log.Printf("remove image file %q: %v", path, err)
	}
}

func (s *server) prepareImages(r *http.Request, messages []requestnorm.Message, ids []int) ([]int, string, error) {
	var sources []string
	for _, message := range messages {
		for _, part := range message.Content.Parts {
			kind, _ := part.Lookup("type")
			name, _ := kind.Text()
			if name == "image" {
				source, _ := part.Lookup("source")
				text, _ := source.Text()
				sources = append(sources, text)
			}
		}
	}
	if len(sources) == 0 {
		return ids, "", nil
	}
	if s.cfg.Vision == nil {
		return nil, "", errors.New("images are not supported")
	}
	origin := r.Header.Get("Origin")
	if origin != "" {
		parsed, err := url.Parse(origin)
		if err != nil || parsed.Host == "" || !strings.EqualFold(parsed.Host, r.Host) {
			for _, source := range sources {
				if !strings.HasPrefix(source, "data:") && !strings.HasPrefix(source, "http://") && !strings.HasPrefix(source, "https://") {
					return nil, "", errors.New("local images are not allowed from a foreign origin")
				}
			}
		}
	}
	start, err := s.cfg.Tokenizer.Encode("<|vision_start|>", true, nil)
	if err != nil || len(start) != 1 {
		return nil, "", errors.New("vision start token is not available")
	}
	pad, err := s.cfg.Tokenizer.Encode("<|image_pad|>", true, nil)
	if err != nil || len(pad) != 1 {
		return nil, "", errors.New("image pad token is not available")
	}
	literal, err := s.cfg.Tokenizer.Encode("<|image_pad|>", false, nil)
	if err != nil || len(literal) == 0 || (len(literal) == 1 && literal[0] == pad[0]) {
		return nil, "", errors.New("invalid image pad token")
	}
	plainStart, err := s.cfg.Tokenizer.Encode("<|vision_start|>", false, nil)
	if err != nil || (len(plainStart) == 1 && plainStart[0] == start[0]) {
		return nil, "", errors.New("invalid vision start token")
	}
	var paths []string
	defer func() {
		for _, path := range paths {
			removeImageFile(path)
		}
	}()
	counts := make([]int, 0, len(sources))
	for _, source := range sources {
		data, err := visioninput.Load(r.Context(), source)
		if err != nil {
			return nil, "", err
		}
		ctx, cancel := context.WithTimeout(r.Context(), 300*time.Second)
		path, count, err := s.cfg.Vision.Encode(ctx, data)
		cancel()
		if err != nil {
			return nil, "", fmt.Errorf("encode image: %w", err)
		}
		paths = append(paths, path)
		counts = append(counts, count)
	}
	ids, err = expandImageTokens(ids, start[0], pad[0], literal, counts, s.cfg.Engine.ContextLimit())
	if err != nil {
		return nil, "", err
	}
	combined, err := visioninput.Combine(paths)
	if err != nil {
		return nil, "", err
	}
	return ids, combined, nil
}
