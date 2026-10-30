package main

import (
	"context"
	"fmt"
	"os"
)

func inheritedOwnerFile(fd int) (*os.File, error) {
	if fd != 0 {
		return nil, fmt.Errorf("--owner-fd must be standard input on Windows")
	}
	return os.Stdin, nil
}

func watchOwner(ctx context.Context, cancel context.CancelFunc, file *os.File) error {
	go readOwner(ctx, file, cancel)
	return nil
}
