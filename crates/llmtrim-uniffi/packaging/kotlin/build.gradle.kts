// Publishable JVM library for the llmtrim Kotlin bindings.
//
// The UniFFI-generated Kotlin (src/main/kotlin/uniffi/…) loads the native engine through
// JNA, which resolves the library from the classpath at /<os-arch>/, so the compiled
// cdylib is bundled under src/main/resources/<os-arch>/ and ends up inside the jar. Both
// the generated sources and the native resources are placed by scripts/build-maven.sh and
// are git-ignored build artifacts. A release jar bundles every platform's library.
//
//   ./gradlew build      # compile + jar
//   ./gradlew publish    # to the configured Maven repo (creds via env)

import com.vanniktech.maven.publish.SonatypeHost

plugins {
    kotlin("jvm") version "2.0.21"
    // Publishes to Maven Central (Central Portal) and handles signing. Credentials/keys
    // come from ORG_GRADLE_PROJECT_* env at publish time; `build`/`jar` work without them.
    id("com.vanniktech.maven.publish") version "0.30.0"
}

group = "io.github.fkiene"
version = System.getenv("LLMTRIM_VERSION") ?: "0.1.7-SNAPSHOT"

repositories { mavenCentral() }

dependencies {
    // The generated bindings call into the native library via JNA. `api` so consumers
    // get it transitively.
    api("net.java.dev.jna:jna:5.14.0")
}

// Target JVM 17 bytecode, compiled by whatever JDK >= 17 runs the build (avoids Gradle
// toolchain auto-detection requiring an exact 17 install).
kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}
java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

mavenPublishing {
    publishToMavenCentral(SonatypeHost.CENTRAL_PORTAL, automaticRelease = true)
    signAllPublications()
    coordinates("io.github.fkiene", "llmtrim", project.version.toString())
    pom {
        name.set("llmtrim")
        description.set("Static, deterministic LLM prompt/payload compression that cuts input tokens 30-90% with zero extra model calls.")
        url.set("https://github.com/fkiene/llmtrim")
        licenses {
            license {
                name.set("MPL-2.0")
                url.set("https://www.mozilla.org/MPL/2.0/")
            }
        }
        developers {
            developer {
                id.set("fkiene")
                name.set("François Kiene")
            }
        }
        scm {
            url.set("https://github.com/fkiene/llmtrim")
            connection.set("scm:git:https://github.com/fkiene/llmtrim.git")
            developerConnection.set("scm:git:ssh://git@github.com/fkiene/llmtrim.git")
        }
    }
}
