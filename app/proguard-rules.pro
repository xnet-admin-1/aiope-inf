# AIOPE-INF: Proguard Rules
# Keep JNI methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep callback interfaces used by JNI
-keep class com.aiope.inf.LlamaJNI$StreamCallback { *; }
-keep class com.aiope.inf.LlamaJNI$TokenCallback { *; }
-keep class com.aiope.inf.LlamaJNI$QuantizeProgressCallback { *; }

# Keep LlamaJNI class
-keep class com.aiope.inf.LlamaJNI { *; }

# Keep LiteRT-LM SDK (JNI callbacks from native)
-keep class com.google.ai.edge.litertlm.** { *; }
