/**
 * Project dignose
 * @author gdyshi
 * @version v0.1
 */

#include "jni_utils.h"
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

/*
 * Get a human-readable summary of an exception object.  The buffer will
 * be populated with the "binary" class name and, if present, the
 * exception message.
 */
static void getExceptionSummary(JNIEnv *env, jthrowable exception, char *buf, size_t bufLen) {
    int success = 0;

    /* get the name of the exception's class */
    jclass exceptionClazz = env->GetObjectClass(exception); // can't fail
    jclass classClazz = env->GetObjectClass(exceptionClazz); // java.lang.Class, can't fail
    jmethodID classGetNameMethod = env->GetMethodID(
            classClazz, "getName", "()Ljava/lang/String;");
    jstring classNameStr = (jstring) env->CallObjectMethod(exceptionClazz, classGetNameMethod);
    if (classNameStr != NULL) {
        /* get printable string */
        const char *classNameChars = env->GetStringUTFChars(classNameStr, NULL);
        if (classNameChars != NULL) {
            /* if the exception has a message string, get that */
            jmethodID throwableGetMessageMethod = env->GetMethodID(
                    exceptionClazz, "getMessage", "()Ljava/lang/String;");
            jstring messageStr = (jstring) env->CallObjectMethod(
                    exception, throwableGetMessageMethod);

            if (messageStr != NULL) {
                const char *messageChars = env->GetStringUTFChars(messageStr, NULL);
                if (messageChars != NULL) {
                    snprintf(buf, bufLen, "%s: %s", classNameChars, messageChars);
                    env->ReleaseStringUTFChars(messageStr, messageChars);
                } else {
                    env->ExceptionClear(); // clear OOM
                    snprintf(buf, bufLen, "%s: <error getting message>", classNameChars);
                }
                env->DeleteLocalRef(messageStr);
            } else {
                strncpy(buf, classNameChars, bufLen);
                buf[bufLen - 1] = '\0';
            }

            env->ReleaseStringUTFChars(classNameStr, classNameChars);
            success = 1;
        }
        env->DeleteLocalRef(classNameStr);
    }
    env->DeleteLocalRef(classClazz);
    env->DeleteLocalRef(exceptionClazz);

    if (!success) {
        env->ExceptionClear();
        snprintf(buf, bufLen, "%s", "<error getting class name>");
    }
}

/*
 * Throw an exception with the specified class and an optional message.
 *
 * If an exception is currently pending, we log a warning message and
 * clear it.
 *
 * Returns 0 if the specified exception was successfully thrown.  (Some
 * sort of exception will always be pending when this returns.)
 */
int jniThrowException(JNIEnv *env, const char *className, const char *msg) {
    jclass exceptionClass;

    if (env->ExceptionCheck()) {
        /* TODO: consider creating the new exception with this as "cause" */
        char buf[256];

        jthrowable exception = env->ExceptionOccurred();
        env->ExceptionClear();

        if (exception != NULL) {
            getExceptionSummary(env, exception, buf, sizeof(buf));
            LOG("Discarding pending exception (%s) to throw %s\n", buf, className);
            env->DeleteLocalRef(exception);
        }
    }

    exceptionClass = env->FindClass(className);
    if (exceptionClass == NULL) {
        LOG("Unable to find exception class %s\n", className);
        /* ClassNotFoundException now pending */
        return -1;
    }

    int result = 0;
    if (env->ThrowNew(exceptionClass, msg) != JNI_OK) {
        LOG("Failed throwing '%s' '%s'\n", className, msg);
        /* an exception, most likely OOM, will now be pending */
        result = -1;
    }

    env->DeleteLocalRef(exceptionClass);
    return result;
}

int find_methods(JNIEnv *env, method *methods, int count) {
    for (int i = 0; i < count; i++) {
        method *m = &methods[i];
        jclass clazz = env->FindClass(m->class_name);
        if (clazz == NULL) {
            LOG("Can't find %s", m->class_name);
            return -1;
        }

        jmethodID mid = env->GetMethodID(clazz, m->method_name, m->method_type);
        if (mid == NULL) {
            LOG("Can't find %s.%s", m->class_name,
                                m->method_name);
            return -1;
        }

        *(m->jmethod) = mid;
    }

    return 0;
}

int find_fields(JNIEnv *env, field *fields, int count) {
    assert(env != NULL);
    assert(fields != NULL);
    for (int i = 0; i < count; i++) {
        field *f = &fields[i];
        jclass clazz = env->FindClass(f->class_name);
        if (clazz == NULL) {
            LOG("find_fields Can't find class %s",
                                f->class_name);
            return -1;
        }

        jfieldID field = env->GetFieldID(clazz, f->field_name, f->field_type);
        if (field == NULL) {
            LOG("find_fields Can't find %s.%s",
                                f->class_name, f->field_name);
            return -1;
        }

        *(f->jfield) = field;
    }

    return 0;
}
