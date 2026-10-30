package tokenizer

const (
	classLetter byte = 1 << iota
	classMark
	classNumber
	classPunctuation
	classSymbol
	classSpace
)

// runeClass uses data generated from the Python reference's regex release.
// In particular, Go 1.22's unicode tables are older than this data.
func runeClass(r rune) byte {
	if r < 0 || r > 0x10ffff {
		return 0
	}
	lo, hi := 0, len(unicodeBoundaries)
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		if unicodeBoundaries[mid]>>6 <= uint32(r) {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return byte(unicodeBoundaries[lo-1] & 63)
}
