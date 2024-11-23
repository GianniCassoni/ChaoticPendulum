CC = g++
CFLAGS = -I ./usr/include
SRC = Fractal_mama_Parallel.cpp
TARGET = Fractal_mama

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -O3 -o $@ $(SRC)

clean:
	rm -f $(TARGET)
