CC = gcc
CFLAGS = -O3 -Wall
LDFLAGS = -lopenblas -lpthread -lm

all: knn

knn: knn.c
	$(CC) $(CFLAGS) -o knn knn.c $(LDFLAGS)

clean:
	rm -f knn