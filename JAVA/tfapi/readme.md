
# 编译
```
javac -cp libtensorflow-1.12.0.jar:guava-28.0-jre.jar JavaModel.java Example.java
```

# 运行
```
java -cp libtensorflow-1.12.0.jar:guava-28.0-jre.jar:. -Djava.library.path=./jni Example
```