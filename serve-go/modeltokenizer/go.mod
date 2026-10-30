module strata/serve-go/modeltokenizer

go 1.22

require (
	strata/serve-go/bpe v0.0.0
	strata/tokenizer v0.0.0
)

replace strata/serve-go/bpe => ../bpe

replace strata/tokenizer => ../tokenizer
