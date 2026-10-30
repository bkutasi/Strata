package server

import "errors"

// expandImageTokens replaces template image placeholders in prompt order.
// A pad token outside a vision-start pair is literal text, not an image.
func expandImageTokens(ids []int, start, pad int, literal []int, counts []int, limit int) ([]int, error) {
	if limit < 0 || len(literal) == 0 {
		return nil, errors.New("invalid image token configuration")
	}
	size, image := 0, 0
	for i, id := range ids {
		n := 1
		if id == pad {
			if i > 0 && ids[i-1] == start {
				if image >= len(counts) || counts[image] <= 0 {
					return nil, errors.New("the prompt and its images do not match")
				}
				n = counts[image]
				image++
			} else {
				n = len(literal)
			}
		}
		if n > limit-size {
			return nil, errors.New("image prompt exceeds the context window")
		}
		size += n
	}
	if image != len(counts) {
		return nil, errors.New("the prompt and its images do not match")
	}
	out := make([]int, 0, size)
	image = 0
	for i, id := range ids {
		if id != pad {
			out = append(out, id)
		} else if i > 0 && ids[i-1] == start {
			for n := 0; n < counts[image]; n++ {
				out = append(out, pad)
			}
			image++
		} else {
			out = append(out, literal...)
		}
	}
	return out, nil
}
