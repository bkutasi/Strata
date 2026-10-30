//go:build unix

package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"syscall"
	"time"
)

func inheritedOwnerFile(fd int) (*os.File, error) {
	if fd < 0 {
		return nil, fmt.Errorf("--owner-fd must be an inherited pipe")
	}
	file := os.NewFile(uintptr(fd), "owner pipe")
	if file == nil {
		return nil, fmt.Errorf("--owner-fd must be an inherited pipe")
	}
	return file, nil
}

func watchOwner(ctx context.Context, cancel context.CancelFunc, file *os.File) error {
	pgid := syscall.Getpgrp()
	if pgid != os.Getpid() {
		return fmt.Errorf("--owner-fd requires a dedicated POSIX server process group")
	}
	go readOwner(ctx, file, func() {
		timer := time.NewTimer(45 * time.Second)
		defer timer.Stop()
		stopOwnerGroup(pgid, timer.C, cancel)
	})
	return nil
}

func stopOwnerGroup(pgid int, deadline <-chan time.Time, cancel context.CancelFunc) {
	if err := syscall.Kill(-pgid, syscall.SIGTERM); err != nil {
		log.Printf("owner process group stop: %v", err)
	}
	cancel()
	<-deadline
	if err := syscall.Kill(-pgid, syscall.SIGKILL); err != nil {
		log.Printf("owner process group kill: %v", err)
	}
}
