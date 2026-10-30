module strata/template

go 1.26

toolchain go1.26.10

require (
	github.com/jryberg/gojinja v1.1.1
	strata/requestnorm v0.0.0
)

replace strata/requestnorm => ../requestnorm
