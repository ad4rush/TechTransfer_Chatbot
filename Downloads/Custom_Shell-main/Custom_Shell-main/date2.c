#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int displayTime = 0; // Option -d
    int rfc5322Format = 0; // Option -R
    const char *filename = NULL;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s [-d] [-R] <filename>\n", argv[0]);
        return 1;
    }

    int argIndex = 1;

    // Check for display time option
    if (strcmp(argv[argIndex], "-d") == 0) {
        displayTime = 1;
        argIndex++;
    }

    // Check for RFC 5322 format option
    if (strcmp(argv[argIndex], "-R") == 0) {
        rfc5322Format = 1;
        argIndex++;
    }

    // Get the filename argument
    if (argIndex < argc) {
        filename = argv[argIndex];
    } else {
        fprintf(stderr, "Usage: %s [-d] [-R] <filename>\n", argv[0]);
        return 1;
    }

    struct stat fileInfo;

    if (stat(filename, &fileInfo) != 0) {
        perror("Error getting file information");
        return 2;
    }

    // Get the last modified time from the file information
    time_t modifiedTime = fileInfo.st_mtime;

    // Convert the time to a human-readable string
    struct tm *timeInfo = localtime(&modifiedTime);
    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeInfo);
    if(!rfc5322Format){
        printf("Last modified time of %s: %s\n", filename, buffer);
    }
    // return 0;
    if(displayTime){
        printf("Last modified time of %s: %s\n", filename, buffer);
    }
    // Convert the time to RFC 5322 format
    char rfc5322[80];
    strftime(rfc5322, sizeof(rfc5322), "%a %d %b %Y %H:%M:%S %z", timeInfo);
    if (rfc5322Format) {
        printf("Last modified time of %s: \033[1mRFC 5322 format\033[0m: %s\n", filename, rfc5322);
    }

    return 0;
}
