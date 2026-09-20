package collective

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/binary"
	"io"
	"math"
	"net"
)

const (
	headerBytes            = 132
	chunkBytes             = 4096
	gradientFrame   uint32 = 1
	parametersFrame uint32 = 2
	ackFrame        uint32 = 3
	commitFrame     uint32 = 4
	anyRank         uint32 = math.MaxUint32
)

var magic = [8]byte{'T', 'A', 'Y', 'I', 'G', 'R', 'D', '1'}

type frame struct {
	rank   uint32
	values []float32
}

func encode(values []float32) []byte {
	payload := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(payload[i*4:], math.Float32bits(value))
	}
	return payload
}

func (c *configuration) header(kind, rank, step uint32, length int) [headerBytes]byte {
	var h [headerBytes]byte
	copy(h[:8], magic[:])
	for i, value := range []uint32{kind, c.spec.World, rank, step, c.spec.Steps, c.spec.Dimension, uint32(length)} {
		binary.LittleEndian.PutUint32(h[8+4*i:], value)
	}
	copy(h[36:68], c.session[:])
	copy(h[68:100], c.contract[:])
	copy(h[100:132], c.layout[:])
	return h
}

func (c *configuration) write(ctx context.Context, conn net.Conn, kind, rank, step uint32, payload []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	header := c.header(kind, rank, step, len(payload))
	auth := hmac.New(sha256.New, c.spec.Secret)
	_, _ = auth.Write(header[:])
	_, _ = auth.Write(payload)
	for _, part := range [][]byte{header[:], payload, auth.Sum(nil)} {
		for len(part) != 0 {
			n, err := conn.Write(part)
			if err != nil {
				return transportError(ctx, err)
			}
			if n <= 0 || n > len(part) {
				return ErrTransport
			}
			part = part[n:]
		}
	}
	return ctx.Err()
}

func (c *configuration) read(ctx context.Context, conn net.Conn, kind, rank, step uint32) (frame, error) {
	var result frame
	var header [headerBytes]byte
	if _, err := io.ReadFull(conn, header[:]); err != nil {
		return result, transportError(ctx, err)
	}
	result.rank = binary.LittleEndian.Uint32(header[16:20])
	length := binary.LittleEndian.Uint32(header[32:36])
	expectedLength := 0
	if kind == gradientFrame || kind == parametersFrame {
		expectedLength = c.payload
	}
	// Reject lengths and all fixed metadata before any peer-sized allocation.
	expected := c.header(kind, result.rank, step, expectedLength)
	if !bytes.Equal(header[:], expected[:]) || result.rank >= c.spec.World ||
		(rank != anyRank && result.rank != rank) || uint64(length)+headerBytes+sha256.Size > c.spec.MaxFrameBytes {
		return frame{}, ErrProtocol
	}
	auth := hmac.New(sha256.New, c.spec.Secret)
	_, _ = auth.Write(header[:])
	if expectedLength != 0 {
		result.values = make([]float32, c.spec.Dimension)
	}
	var chunk [chunkBytes]byte
	invalid := false
	for at := 0; at < expectedLength; {
		if err := ctx.Err(); err != nil {
			return frame{}, err
		}
		part := chunk[:min(len(chunk), expectedLength-at)]
		if _, err := io.ReadFull(conn, part); err != nil {
			return frame{}, transportError(ctx, err)
		}
		_, _ = auth.Write(part)
		for i := 0; i < len(part); i += 4 {
			value := math.Float32frombits(binary.LittleEndian.Uint32(part[i:]))
			invalid = invalid || math.IsNaN(float64(value)) || math.IsInf(float64(value), 0)
			result.values[(at+i)/4] = value
		}
		at += len(part)
	}
	var signature [sha256.Size]byte
	if _, err := io.ReadFull(conn, signature[:]); err != nil {
		return frame{}, transportError(ctx, err)
	}
	if !hmac.Equal(signature[:], auth.Sum(nil)) {
		return frame{}, ErrAuth
	}
	if invalid {
		return frame{}, ErrNonFinite
	}
	return result, ctx.Err()
}
