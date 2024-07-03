#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>

int word_counter(FILE *file, int flagg){
    // Count words in the file
    int word_count = 0;
    char ch;
    int flag = 0;
    int newlineflag=0;
    while ((ch = fgetc(file)) != EOF) {
        //printf("%d flag%d %d %d\n",ch,flag,flagg,newlineflag);
        if(!flagg){
        if ((ch == 13)){
            //printf("YES");
            newlineflag=1;
            continue;
        }
        else if (newlineflag==1 && ch==10){
            newlineflag=0;
            if(!flagg){
                //printf("YESS");
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
            newlineflag=0;
            // printf("%c",ch);
            continue;
        }
        else{
            flag=1;
        }
        newlineflag=0;
    }
    // printf("%d",flag);
   
    return word_count+flag;
}
int extra(const char *arg){
    if (arg) {
        char *extra_arg = strtok(NULL, " ");
        if (extra_arg) {
            return 0;
        } else {
            return 1;
        }
    } else {
        printf("Usage: word [-n] <filename>\n");
    }
    return 0;
}
int execute_word(const char *arg) {
    printf("Executing 'word' command with argument: %s\n", arg);
    // if (argc < 2|| argc > 4) {
    //     fprintf(stderr, "Too many/less Arguments received \n2 =< Needed =< 4\n");
    //     return 1;
    // }
    FILE *file = fopen(arg, "r");
    if (file == NULL) {
        perror("Error");
        return 1;
    }
    printf("Number of word in file named %s is %d",arg,word_counter(file,1));
    return 0;
    
}
int execute_word_n(const char *arg) {
    printf("Executing 'word -n' command with argument: %s\n", arg);

    // Add code here to execute word command (from word.c) with arg and removeIfExists
    // if (argc < 2|| argc > 4) {
    //     fprintf(stderr, "Too many/less Arguments received \n2 =< Needed =< 4\n");
    //     return 1;
    // }
    FILE *file = fopen(arg, "r");
    if (file == NULL) {
        perror("Error");
        return 1;
    }
    printf("Number of word in file named %s is %d",arg,word_counter(file,1));
    return 0;
    
}
int execute_word_d(const char *arg, const char *arg2) {
    printf("Executing 'word -d' command with argument: %s\n", arg);

    // Add code here to execute word command (from word.c) with arg and removeIfExists
    // if (argc < 2|| argc > 4) {
    //     fprintf(stderr, "Too many/less Arguments received \n2 =< Needed =< 4\n");
    //     return 1;
    // }
    // if (n) {
    //     FILE *file = fopen(arg, "r");
    //     if (file == NULL) {
    //         perror("Error");
    //         return 1;}
    //     printf("Number of word in file named %s is %d",arg,word_counter(file,1));
    //     return 0;
    // }
    // if (argc == 3) {
    // FILE *file = fopen(arg, "r");
    // if (file == NULL) {
    //     perror("Error");
    //     return 1;}
    // printf("Number of word in file named %s is %d",argv[2],word_counter(file,0));
    // return 0;
    // }
    // int ignore_newline = 0;
    // int calc_difference = 0;

    // // Parse options
    
    // if (strcmp(argv[1], "-n") == 0) {
    //     ignore_newline = 1;
    // } else if (strcmp(argv[1], "-d") == 0) {
    //     calc_difference = 1;
    // }

    // // Open the file

    

    

    // if (argc == 4) {
        // Open the second file
    FILE *file2 = fopen(arg2, "r");
    FILE *file1 = fopen(arg, "r");
    if ((file1 == NULL)|(file2 == NULL)) {
        perror("Error");
        fclose(file1);
        return 1;
    }
    int s1=word_counter(file1,1);
    int s2=word_counter(file2,1);
    printf("Word count of %s: %d\nWord count of %s: %d\n\n\e[1mDifference in word count of %s and %s file is: %d\e[m\n", arg, s1, arg2, s2, arg, arg2, abs(s1-s2));

    fclose(file2);
    fclose(file1);
    return 0;
    // fprintf(stderr, "Usage: word -n file_name | Usage: word -d file_name file_name2\n");
    // printf("Word count: %d\n", word_count);

    // if (calc_difference) {
    //     printf("Difference in word counts: %d\n", abs(word_count ));
    // }

    // fclose(file);
    // return 0;

}

int execute_dir(const char *arg, int r1, int r2, int v1, int v2) {
    int pid = fork();

    if (pid == -1) {
        perror("fork");
        return 1;
    } else if (pid == 0) {
        // Child process
        if (r1 + r2 + v1 + v2 == 0) {
            char arg_copy[strlen(arg) + 1];
            strcpy(arg_copy, arg);
            char *args[] = {"dir", arg_copy, NULL};
            execvp("./dir",args);
            exit(1);
        }
        else if((r1==0&&v1==1)||(r1+v2==2)||(r2+v1==2)){
            char arg_copy[strlen(arg) + 1];
            strcpy(arg_copy, arg);
            char *args[] = {"dir", "-v", arg_copy, NULL};
            execvp("./dir",args);
            exit(1);
        }
        else if(r1==1){
            char arg_copy[strlen(arg) + 1];
            strcpy(arg_copy, arg);
            char *args[] = {"dir", "-r", arg_copy, NULL};
            execvp("./dir",args);
            exit(1);
        }

        // Handle other cases if needed
        exit(0);
    } else {
        // Parent process
        int status;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        } else {
            return 1; // Child process didn't exit normally
        }
    }
}
int dir_remove(const char *arg){
    int pid = fork();

    if (pid == -1) {
        perror("fork");
        return 1;
    } else if (pid == 0) {
        
        char arg_copy[strlen(arg) + 1];
        strcpy(arg_copy, arg);
        char *args[] = {"dir", "-rr", arg_copy, NULL};
        execvp("./dir",args);
        exit(1);
    }
    else {
        // Parent process
        int status;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        } else {
            return 1; // Child process didn't exit normally
        }
    }
        
        exit(0);

}
int execute_date_R(const char *arg){
    int pid = fork();

    if (pid == -1) {
        perror("fork");
        return 1;
    } else if (pid == 0) {
        
        char arg_copy[strlen(arg) + 1];
        strcpy(arg_copy, arg);
        char *args[] = {"date", "-R", arg_copy, NULL};
        execvp("./date",args);
        exit(1);
    }else {
        // Parent process
        int status;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        } else {
            return 1; // Child process didn't exit normally
        }
    }
    exit(0);

}
int execute_date_Rd(const char *arg){
    int pid = fork();

    if (pid == -1) {
        perror("fork");
        return 1;
    } else if (pid == 0) {
        
        char arg_copy[strlen(arg) + 1];
        strcpy(arg_copy, arg);
        char *args[] = {"date", "-R","-d",arg_copy, NULL};
        execvp("./date",args);
        exit(1);
    }else {
        // Parent process
        int status;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        } else {
            return 1; // Child process didn't exit normally
        }
    }
    exit(0);
}
int execute_date_d(const char *arg){
    int pid = fork();

    if (pid == -1) {
        perror("fork");
        return 1;
    } else if (pid == 0) {
        
        char arg_copy[strlen(arg) + 1];
        strcpy(arg_copy, arg);
        char *args[] = {"date", "-d",arg_copy, NULL};
        execvp("./date",args);
        exit(1);
    }else {
        // Parent process
        int status;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        } else {
            return 1; // Child process didn't exit normally
        }
    }
    exit(0);
}
int execute_date(const char *arg){
    int pid = fork();
    // printf("HI");
    if (pid == -1) {
        perror("fork");
        return 1;
    } else if (pid == 0) {
        
        char arg_copy[strlen(arg) + 1];
        strcpy(arg_copy, arg);
        char *args[] = {"date", arg_copy, NULL};
        execvp("./date",args);
        exit(1);
    }else {
        // Parent process
        int status;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        } else {
            return 1; // Child process didn't exit normally
        }
    }
    exit(0);

}


// void handle_ctrl_c(int sig) {
//     // Handle Ctrl+C (SIGINT)
//     printf("\nShell terminated.\n");
//     exit(0);
// }

int main() {
    // signal(SIGINT, handle_ctrl_c); // Register Ctrl+C signal handler

    char input[1024];
    printf("\n\033[1m\x1b[36mAmartya's Shell started\x1b[0m\033[0m\n");
    while (1) {
        char cwd2[1024];
        char *username = getenv("USER");
            if (username == NULL) {
                perror("getenv");
                return 1;
            }

        char hostname[256];
        if (gethostname(hostname, sizeof(hostname)) != 0) {
            perror("gethostname");
            return 1;
        }

        if (getcwd(cwd2, sizeof(cwd2)) != NULL) {
            //printf("$");
        printf("\n\033[1m\033[32m%s@%s\033[0m:\033[1m\033[34m%s\033[0m\033[0m$ ",username,hostname,cwd2);
        sleep(1);
        }
        // printf("Username: %s\n", username);
        // printf("Hostname: %s\n", hostname);
        fgets(input, sizeof(input), stdin);
        input[strcspn(input, "\n")] = '\0'; // Remove trailing newline

        if ((strcmp(input, "exit") == 0)||(strcmp(input, "end") == 0)) {
            printf("Shell terminated.\n");
            break;
        } else {
            char *command = strtok(input, " ");
            char *arg = strtok(NULL, " ");
            if (command) {
                if (strcmp(command, "end") == 0){
                    break;
                }}
            if (command) {
                if (strcmp(command, "word") == 0) {
                    // int n = 0;
                    // int d = 0;
                    if ((arg && strcmp(arg, "-n") == 0)) {
                        // n = 1;
                        arg = strtok(NULL, " ");
                         // Move to the next argument
                        if (arg) {
                            char *extra_arg = strtok(NULL, " ");
                            if (extra_arg) {
                                printf("Error: Too many arguments for 'word' command.\n");
                                continue;
                            } else {
                                execute_word_n(arg);
                                continue;
                            }
                        } 
                        else {
                            printf("Usage: word [-n] <filename>\n");
                            continue;
                        }
                    }
                    else if ((arg && strcmp(arg, "-d") == 0)) {
                        // d = 1;
                        arg = strtok(NULL, " ");
                        char *arg2;
                        arg2 = strtok(NULL, " ");
                        if (arg2) {
                            char *extra_arg = strtok(NULL, " ");
                            if (extra_arg) {
                                printf("Error: Too many arguments for 'word' command.\n");
                                continue;
                            } else {
                                execute_word_d(arg, arg2);
                                continue;// Move to the next argument
                            }
                        } 
                        else {
                            printf("Usage: word [-d] <filename>\n");
                            continue;
                        }

                    }
                    char *extra_arg = strtok(NULL, " ");
                    if (!extra_arg || strlen(arg)==0){
                        execute_word(arg);
                    }
                    else{
                        printf("Erorr:The command is incorrect");
                    }
                    
                    
                } 
                else if (strcmp(command, "date") == 0) {
                    int d = 0;
                    int R = 0;
                    if ((arg && strcmp(arg, "-d") == 0)) {
                        d = 1;
                        arg = strtok(NULL, " ");
                        if ((arg && strcmp(arg, "-R")==0)){
                            R=1;
                            arg = strtok(NULL, " ");
                        }
                         // Move to the next argument
                        if (arg) {
                            char *extra_arg = strtok(NULL, " ");
                            if (extra_arg) {
                                printf("Error: Too many arguments for 'date' command.\n");
                                continue;
                            } else {
                                if(d+R==2){
                                    execute_date_Rd(arg);
                                    continue;
                                }
                                else{
                                execute_date_d(arg);
                                continue;
                                }
                            }
                        } 
                        else {
                            printf("Usage: date [-d] <filename>\n");
                            continue;
                        }
                    }
                    else if ((arg && strcmp(arg, "-R") == 0)) {
                        R = 1;
                        arg = strtok(NULL, " ");
                        if ((arg && strcmp(arg, "-d")==0)){
                            d=1;
                            arg = strtok(NULL, " ");
                        }
                         // Move to the next argument
                        if (arg) {
                            char *extra_arg = strtok(NULL, " ");
                            if (extra_arg) {
                                printf("Error: Too many arguments for 'date' command.\n");
                                continue;
                            } else {
                                if(d+R==2){
                                    execute_date_Rd(arg);
                                    continue;
                                }
                                else{
                                execute_date_R(arg);
                                continue;
                                }
                            }
                        } 
                        else {
                            printf("Usage: date [-R] <filename>\n");
                            continue;
                        }
                    }
                    char *extra_arg = strtok(NULL, " ");
                    if (!extra_arg){
                        execute_date(arg);
                        continue;
                    }
                    else{
                        printf("Erorr:The command is incorrect");
                    }
                    

                }
                else if (strcmp(command, "dir") == 0) {
                    int r1 = 0;
                    int r2 = 0;
                    int v1 = 0;
                    int v2 = 0;
                    if ((arg && strcmp(arg, "-rr") == 0)) {
                        arg = strtok(NULL, " "); // Move to the next argument
                        dir_remove(arg);
                        continue;
                    }
                    if ((arg && strcmp(arg, "-r") == 0)) {
                        r1 = 1;
                        arg = strtok(NULL, " "); // Move to the next argument
                        if ((arg && strcmp(arg, "-v") == 0)) {
                        v2 = 1;
                        arg = strtok(NULL, " "); // Move to the next argument
                    }
                    }
                    else if ((arg && strcmp(arg, "-v") == 0)) {
                        v1 = 1;
                        arg = strtok(NULL, " "); // Move to the next argument
                        if ((arg && strcmp(arg, "-r") == 0)) {
                        r2 = 1;
                        arg = strtok(NULL, " "); // Move to the next argument
                    }
                    }

                    
                    if (arg) {
                        char *extra_arg = strtok(NULL, " ");
                        if (extra_arg) {
                            printf("Error: Too many arguments for 'dir' command.\n");
                            continue;
                        } else {
                            execute_dir(arg, r1, r2, v1, v2);
                            printf("%d%d%d%d",r1,r2,v1,v2);
                            chdir(arg);
                            continue;
                        }
                    } else {
                        printf("Usage: dir [-r] <dirname>\n");
                        continue;
                    }
                } 
                else {
                    printf("Unknown command: %s\n", command);
                    continue;
                }
            
        }}
    }
    printf("\033[1m\033[31mAmartya's Shell Ended\033[31m\033[0m\n\n");
    return 0;
}
