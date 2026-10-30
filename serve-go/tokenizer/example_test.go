package tokenizer_test

import (
	"fmt"

	"strata/tokenizer"
)

func ExampleQwen35() {
	text := "Hi 你好12!"
	spans, err := tokenizer.Qwen35(text)
	if err != nil {
		panic(err)
	}
	for _, span := range spans {
		fmt.Printf("%d:%d %q\n", span.Start, span.End, text[span.Start:span.End])
	}
	// Output:
	// 0:2 "Hi"
	// 2:9 " 你好"
	// 9:10 "1"
	// 10:11 "2"
	// 11:12 "!"
}

func ExampleDeepSeek3() {
	text := "Hi 你好12!"
	spans, err := tokenizer.DeepSeek3(text)
	if err != nil {
		panic(err)
	}
	for _, span := range spans {
		fmt.Printf("%d:%d %q\n", span.Start, span.End, text[span.Start:span.End])
	}
	// Output:
	// 0:2 "Hi"
	// 2:3 " "
	// 3:9 "你好"
	// 9:11 "12"
	// 11:12 "!"
}
