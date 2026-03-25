CC      = gcc
CFLAGS  = -O2 -mavx -Wall -Wextra
LDFLAGS = -lm
TARGET  = fft_bench

all: $(TARGET)

$(TARGET): main.c fft.h
	$(CC) $(CFLAGS) -o $(TARGET) main.c $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
