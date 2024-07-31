#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "whisper-encoder.h"
#import "whisper-encoder-impl.h"

#import <CoreML/CoreML.h>

#include <stdlib.h>

#if __cplusplus
extern "C" {
#endif

struct whisper_coreml_context {
    const void * data;
};

struct whisper_coreml_context * whisper_coreml_init(const char * path_model) {
    NSString * path_model_str = [[NSString alloc] initWithUTF8String:path_model];

    NSURL * url_model = [NSURL fileURLWithPath: path_model_str];

    // select which device to run the Core ML model on
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    // config.computeUnits = MLComputeUnitsCPUAndGPU;
    //config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    config.computeUnits = MLComputeUnitsAll;

    const void * data = CFBridgingRetain([[whisper_encoder_impl alloc] initWithContentsOfURL:url_model configuration:config error:nil]);

    if (data == NULL) {
        return NULL;
    }

    whisper_coreml_context * ctx = new whisper_coreml_context;

    ctx->data = data;

    return ctx;
}

void whisper_coreml_free(struct whisper_coreml_context * ctx) {
    CFRelease(ctx->data);
    delete ctx;
}

void whisper_coreml_encode(
        const whisper_coreml_context * ctx,
                             int64_t   n_ctx,
                             int64_t   n_mel,
                               float * mel,
                               float * out) {
    MLMultiArray * inMultiArray = [
        [MLMultiArray alloc] initWithDataPointer: mel
                                           shape: @[@1, @(n_mel), @(n_ctx)]
                                        dataType: MLMultiArrayDataTypeFloat32
                                         strides: @[@(n_ctx*n_mel), @(n_ctx), @1]
                                     deallocator: nil
                                           error: nil
    ];

    @autoreleasepool {
        whisper_encoder_implOutput * outCoreML = [(__bridge id) ctx->data predictionFromLogmel_data:inMultiArray error:nil];

        memcpy(out, outCoreML.output.dataPointer, outCoreML.output.count * sizeof(float));
    }
}

int whisper_coreml_compile(const char * path_model, const char * compile_path_model)
{
    if( path_model == nullptr || compile_path_model == nullptr)
    {
        return -1;
    }
    @autoreleasepool
    {
        @try {
            NSError* error = nil;
            NSURL *modelURL = [NSURL fileURLWithPath:@(path_model)];
            if(![[NSFileManager defaultManager] fileExistsAtPath:@(path_model)])
            {
                return -5;
            }
            NSURL *destinationURL = [NSURL fileURLWithPath:@(compile_path_model)];
            NSURL *compiledURL = [MLModel compileModelAtURL:modelURL error:&error];
            // copy compiled model files to destinationURL and clean up files which producted by CoreML.
            // remove old files
            [[NSFileManager defaultManager] removeItemAtURL:destinationURL error:nil];
            [[NSFileManager defaultManager] copyItemAtURL:compiledURL toURL:destinationURL error:&error];
            [[NSFileManager defaultManager] removeItemAtURL:compiledURL error:nil];
            if(error != nil)
            {
                return -(int)error.code;
            }
        }
        @catch (NSException *exception)
        {
            fprintf(stderr, "%s: exception occurs.", __func__);
            return -6;
        }
    }
    return 0;
}

#if __cplusplus
}
#endif
