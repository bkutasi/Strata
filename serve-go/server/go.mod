module strata/server

go 1.26

require (
	strata.local/httpguard v0.0.0
	strata/engine v0.0.0
	strata/outputparser v0.0.0
	strata/requestnorm v0.0.0
	strata/responses v0.0.0
	strata/serve-go/modeltokenizer v0.0.0
	strata/streamtext v0.0.0
	strata/template v0.0.0
)

require (
	github.com/jryberg/gojinja v1.1.1 // indirect
	strata/serve-go/bpe v0.0.0 // indirect
	strata/tokenizer v0.0.0 // indirect
	strata/toolparser v0.0.0 // indirect
)

replace strata/engine => ../engine

replace strata/serve-go/modeltokenizer => ../modeltokenizer

replace strata/serve-go/bpe => ../bpe

replace strata/tokenizer => ../tokenizer

replace strata/requestnorm => ../requestnorm

replace strata/responses => ../responses

replace strata/outputparser => ../outputparser

replace strata/toolparser => ../toolparser

replace strata/streamtext => ../streamtext

replace strata.local/httpguard => ../httpguard

replace strata/template => ../template
