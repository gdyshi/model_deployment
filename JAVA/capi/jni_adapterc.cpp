//
// Created by guohuihun on 2016/10/17.
//

#include <stddef.h>
#include <assert.h>
#include <jni.h>
#include "jni_utils.h"
#include "model.h"

#define JNI_ADAPTER_CLASS "JavaModel"

// Mutex mLock;
static JavaVM *gJavaVM = NULL;
//static method java_methods_to_find[] = {};

JNIEnv *getJNIEnv() {
    JNIEnv *env = NULL;
    JavaVM *vm = gJavaVM;
    assert(vm != NULL);

    if (vm->AttachCurrentThread((void **)&env, NULL) != JNI_OK) {
        return NULL;
    }
    return env;
}

jint jni_model_init(JNIEnv *env, jobject thiz, jstring j_model_file) {
    int ret = 0;
    char *model_file = (char *) env->GetStringUTFChars(j_model_file, 0);

    ret = model_init(model_file);
    return ret;
}

jint jni_model_deinit(JNIEnv *env, jobject thiz) {
    int ret = 0;
    ret = model_deinit();
    return ret;
}

jint jni_model_inference(JNIEnv *env, jobject thiz, jint batch_size, jfloatArray j_input_vals, jfloatArray j_output_vals) {
    int ret = 0;
    float *input_vals = (float *) (*env).GetFloatArrayElements(j_input_vals, NULL);
    float *output_vals = (float *) (*env).GetFloatArrayElements(j_output_vals, NULL);

    ret = model_inference(batch_size, input_vals, output_vals);

    (*env).ReleaseFloatArrayElements(j_input_vals, (jfloat *) input_vals, 0);
    (*env).ReleaseFloatArrayElements(j_output_vals, (jfloat *) output_vals, 0);
    return ret;
}
static JNINativeMethod gjniDiagnoseMethods[] = {
        {(char *)"model_init",          (char *)"(Ljava/lang/String;)I",            (void *) jni_model_init},
        {(char *)"model_deinit",        (char *)"()I",                              (void *) jni_model_deinit},
        {(char *)"model_inference",     (char *)"(I[F[F)I",                         (void *) jni_model_inference},
};


static int registerJniNativesMethods(JNIEnv *env) {
    jclass clazz;
    static const char *const kClassName = JNI_ADAPTER_CLASS;
    clazz = env->FindClass(kClassName);

    if (clazz == NULL) {
        return -1;
    }

    if (env->RegisterNatives(clazz, gjniDiagnoseMethods,
                             sizeof(gjniDiagnoseMethods) / sizeof(gjniDiagnoseMethods[0])) !=
        JNI_OK) {
        return -1;
    }

    return 0;
}


jint JNI_OnLoad(JavaVM *vm, void *unused) {
    JNIEnv *env = NULL;
    jint result = -1;
    unused = unused;

    if (vm->GetEnv((void **) &env, JNI_VERSION_1_4) != JNI_OK) {
        goto bail;
    }

    gJavaVM = vm;

    assert(env != NULL);

    if (registerJniNativesMethods(env) < 0) {
        goto bail;
    }
//    if (find_methods(env, java_methods_to_find, NELEM(java_methods_to_find)) < 0) {
//        goto bail;
//    }
    result = JNI_VERSION_1_4;
bail:
    return result;
}
