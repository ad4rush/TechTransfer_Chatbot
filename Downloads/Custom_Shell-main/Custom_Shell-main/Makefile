CC = gcc -g
#CC = gcc
CFLAGS = -Wall -Wextra
#CFLAGS = -Wall 


all: dir date shell shell2 word

dir: dir.c
	$(CC) $(CFLAGS) -o $@ $<

date: date.c
	$(CC) $(CFLAGS) -o $@ $<
shell: shell.c
	$(CC) $(CFLAGS) -o $@ $<
shell2: shell2.c
	$(CC) $(CFLAGS) -o $@ $<

word: word.c
	$(CC) $(CFLAGS) -o $@ $<
#use "make clean"
clean:
	rm -f dir date shell2 word
