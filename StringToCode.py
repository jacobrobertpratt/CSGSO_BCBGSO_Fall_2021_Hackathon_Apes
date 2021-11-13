import subprocess
import os
import time

testString = 'int lame = 0; int zoo = 10; for (int i = 0; i < zoo; i++) { lame++; System.out.println("ape" + lame); }'
beginning = "public class MainMethod { public static void main(String[] args) { "
end = "}}"
combined = beginning + testString + end

def run(program):
    tempFile = open("MainMethod.java", 'w')
    tempFile.write(program)
    tempFile.close()
    compile()
    time.sleep(.5)
    start()

def compile():
    command = 'javac ' + "MainMethod.java"
    process = subprocess.Popen(command, shell=True)

def start():
    cmd = 'java ' + "MainMethod"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    suboutput = process.stdout.read()
    print(suboutput.decode("utf-8"))

os.remove('MainMethod.class')
run(combined)
