package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
)

func startOwnerGuard(ctx context.Context, cancel context.CancelFunc, fd int) (*os.File, error) {
	file, err := inheritedOwnerFile(fd)
	if err != nil {
		return nil, err
	}
	info, err := file.Stat()
	if err != nil || info.Mode()&os.ModeNamedPipe == 0 {
		return nil, fmt.Errorf("--owner-fd must be an inherited pipe")
	}
	if err := watchOwner(ctx, cancel, file); err != nil {
		return nil, err
	}
	return file, nil
}

func readOwner(ctx context.Context, file *os.File, onLoss func()) {
	_, err := io.Copy(io.Discard, file)
	if ctx.Err() != nil {
		return
	}
	if err != nil {
		log.Printf("owner pipe: %v", err)
	}
	onLoss()
}
