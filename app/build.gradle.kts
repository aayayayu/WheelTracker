plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

android {
    namespace = "com.example.wheeltracker"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.wheeltracker"
        minSdk = 29
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        ndk {
            // arm64-v8a = real phones, x86_64 = emulator
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

chaquopy {
    defaultConfig {
        // OpenCV for Android is only published for Python 3.8 and 3.10
        version = "3.10"
        pip {
            install("flask")
            install("numpy")
            install("matplotlib")
            install("opencv-python-headless==4.5.1.48")
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-ktx:1.9.3")
    // FFmpeg for Android (community fork; original ffmpeg-kit was removed from Maven Central)
    implementation("com.moizhassan.ffmpeg:ffmpeg-kit-16kb:6.1.1")
    implementation("com.arthenica:smart-exception-java:0.2.1")
}
