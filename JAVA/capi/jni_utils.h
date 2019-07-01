#ifndef PANDACHAT_LAN_JNI_UTILS_H
#define PANDACHAT_LAN_JNI_UTILS_H

#include <jni.h>

#define LOG   printf

#ifndef NELEM
# define NELEM(x) ((int) (sizeof(x) / sizeof((x)[0])))
#endif


int jniThrowException(JNIEnv *env, const char *className, const char *msg);

struct method {
    const char * class_name;
    const char * method_name;
    const char * method_type;
    jmethodID *jmethod;
};

typedef struct field {
    const char * class_name;
    const char * field_name;
    const char * field_type;
    jfieldID *jfield;
} field;

int find_methods(JNIEnv *env, method *methods, int count);

int find_fields(JNIEnv *env, field *fields, int count);


#endif // PANDACHAT_LAN_JNI_UTILS_H
