#include <stdio.h>
#include <string.h>
#include <stdlib.h>
int word_counter(FILE *file, int flagg){
    // Count words in the file
    int word_count = 0;
    char ch;
    int flag = 0;
    int newlineflag=0;
    while ((ch = fgetc(file)) != EOF) {
        printf("%d flag%d %d %d\n",ch,flag,flagg,newlineflag);
        if(!flagg){
        if ((ch == 13)){
            printf("YES");
            newlineflag=1;
            continue;
        }
        else if (newlineflag==1 && ch==10){
            newlineflag=0;
            if(!flagg){
                printf("YESS");
                continue;
            }
        }
        }
        if (ch == ' ' || ch == '\t'|| ((ch == '\n')&&flagg)) {
            if (flag) {
                flag = 0;
                word_count++;

            }
        } 
        else if(newlineflag==1){
            printf("%c",ch);
        }
        else{
            flag=1;
        }
        newlineflag=0;
    }
    printf("%d",flag);
   
    return word_count+flag;
}
int main(int argc, char *argv[]) {
    if (argc < 2|| argc > 4) {
        fprintf(stderr, "Too many/less Arguments received \n2 =< Needed =< 4\n");
        return 1;
    }
    if (argc == 2) {
        FILE *file = fopen(argv[1], "r");
        if (file == NULL) {
            perror("Error");
            return 1;}
        printf("Number of word in file named %s is %d",argv[1],word_counter(file,1));
        return 0;
    }
    if (argc == 3) {
        FILE *file = fopen(argv[2], "r");
        if (file == NULL) {
            perror("Error");
            return 1;}
        printf("Number of word in file named %s is %d",argv[2],word_counter(file,0));
        return 0;
    }
    // int ignore_newline = 0;
    // int calc_difference = 0;

    // // Parse options
    
    // if (strcmp(argv[1], "-n") == 0) {
    //     ignore_newline = 1;
    // } else if (strcmp(argv[1], "-d") == 0) {
    //     calc_difference = 1;
    // }

    // // Open the file

    

    

    if (argc == 4) {
        // Open the second file
        FILE *file2 = fopen(argv[3], "r");
        FILE *file1 = fopen(argv[2], "r");
        if ((file1 == NULL)|(file2 == NULL)) {
            perror("Error");
            fclose(file1);
            return 1;
        }
        int s1=word_counter(file1,1);
        int s2=word_counter(file2,1);
        printf("Word count of %s: %d\nWord count of %s: %d\n\n\e[1mDifference in word count of %s and %s file is: %d\e[m\n", argv[2], s1, argv[3], s2, argv[2], argv[3], abs(s1-s2));

        fclose(file2);
        fclose(file1);
        return 0;
    }
    fprintf(stderr, "Usage: %s -n file_name | Usage: %s -d file_name file_name2\n", argv[0],argv[0]);
    // printf("Word count: %d\n", word_count);

    // if (calc_difference) {
    //     printf("Difference in word counts: %d\n", abs(word_count ));
    // }

    // fclose(file);
    // return 0;
}
