#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
void deleteDirectoryRecursively(const char *dirPath) {
    struct dirent *entry;
    DIR *dir = opendir(dirPath);

    if (dir == NULL) {
        // perror("opendir");
        return;
    }

    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        char fullPath[1024];
        snprintf(fullPath, sizeof(fullPath), "%s/%s", dirPath, entry->d_name);

        if (entry->d_type == DT_DIR) {
            deleteDirectoryRecursively(fullPath);
        } else {
            if (remove(fullPath) != 0) {
                perror("remove");
            }
        }
    }

    closedir(dir);
    if (rmdir(dirPath) != 0) {
        perror("rmdir");
    }
}
int main(int argc, char *argv[]) {
    

    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) == NULL) {
        perror("getcwd");
        return EXIT_FAILURE;
    }
    // printf("%d%s",argc,argv[1]);
    if (argc < 2||argc > 3) {
        fprintf(stderr, "Usage: %s <directory_name>\n", argv[0]);
        return 1;
    }
    if (argc == 2){
    char *dir_name = argv[argc-1];
        DIR *dir = opendir(dir_name);
        if (dir) {
            closedir(dir);
            printf("Directory '%s' already exists.\n", dir_name);
            return 1;
        }

    // Create the directory
    // if (mkdir(argv[1], 0755) == -1) {
    //     perror("Error creating directory");
    //     return 1;
    // }

    // Print a success message
    // printf("Directory '%s' created.\n", argv[1]);
    }
    if (argc == 3){
        // printf("%d%s",argc,argv[1]);
        if(strcmp(argv[1], "-r") == 0){
            // char *dir_name = argv[argc-1];
                // DIR *dir = opendir(dir_name);
                // if (dir) {
                //     closedir(dir);
                //     // printf("Directory '%s' already exists. \n", dir_name);
                //     // printf("Trying to remove........\n");
                    deleteDirectoryRecursively(argv[argc-1]);
                    

                // }
            // if (mkdir(argv[1], 0755) == -1) {
            //     perror("Error creating directory");
            //     return 1;
            // }
        }
        else if(strcmp(argv[1], "-v") == 0){
            char *dir_name = argv[argc-1];

            printf("Current working directory: %s\n", cwd);
            sleep(1);
            printf("\e[1mStep 1:-\e[mChecking if Directory '%s' already exists. \n", dir_name);
            sleep(1);
                DIR *dir = opendir(dir_name);
                if (dir) {
                    closedir(dir);
                    printf("Directory '%s' already exists. \n", dir_name);
                    sleep(1);
                    printf("\n\033[1mStep 2:-\033[0mtrying to remove........\n");
                    sleep(1);
                    deleteDirectoryRecursively(argv[argc-1]);
                    printf("Directory '%s' removed successfully. \n", dir_name);
                    sleep(1);
                    printf("Trying to make '%s' named Directory with read (4), write (2), and execute (1) permissions. . \n", dir_name);
                    sleep(1);
                }
                else{
                    sleep(1);
                    printf("Directory of name '%s' Doesn't exist exists. \n", dir_name);
                    sleep(1);
                    printf("Trying to make '%s' named Directory with read (4), write (2), and execute (1) permissions. . \n", dir_name);
                }
            // if (mkdir(argv[1], 0755) == -1) {
            //     perror("Error creating directory");
            //     return 1;
            // }
        }
        else if(strcmp(argv[1], "-rr") == 0){
            deleteDirectoryRecursively(argv[2]);
            return 0;
        }




    // Create the directory

    // Print a success message
    }
    if (mkdir(argv[argc-1], 0777) == -1) {
        perror("Error creating directory");
        return 1;
    
    }
    if (chdir(argv[argc-1]) == -1) {
        perror("chdir");
        return 1;
    }
    if(strcmp(argv[1], "-v") == 0){
        printf("Directory '%s' created successfully.\n", argv[argc-1]);
        sleep(1);
        printf("Changed to directory: %s\n", argv[argc-1]);
        sleep(1);
    
    char cwd2[1024];
    if (getcwd(cwd2, sizeof(cwd2)) != NULL) {
        printf("Current working directory: %s\n\033[0m", cwd2);
        sleep(1);
    } else {
        perror("getcwd");
        return 1;
    }
    }
    return 0;
}