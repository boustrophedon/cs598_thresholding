CC=g++
CFLAGS=-Wall -Werror -std=c++11
LDFLAGS=-lopencv_core -lopencv_highgui -lopencv_imgproc

all: threshold

OBJS = threshold.o

threshold: $(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS) 

%.o: %.cpp
	$(CC) -o $@ $(CFLAGS) -c $<

clean:
	rm threshold $(OBJS)
