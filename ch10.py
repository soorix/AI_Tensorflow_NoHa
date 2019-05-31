
#현재 실행중인 디렉토리에 파일 생성
f=open("file1.txt","w")

for i in range(1,11):
    data="%d번째 줄입니다.\n"%i
    f.write(data)
f.close()

f=open("file1.txt","r")
while True:
    line = f.readline()
    if not line: break
    print(line, end=" ")
f.close()

#csv파일 형식으로 저장하기. (데이터가 ','로 구분되는 데이터)
import sys
nums = [0,1,2,3,4,5,6,7,8,9]
count = len(nums)
output_file="C:/Users/student/PycharmProjects/untitled3/chapter4/result.csv"
# a = append
f = open(output_file,'a')

for idx in range(count):
    if idx<(count-1):
        f.write(str(nums[idx])+',')
    else:
        f.write(str(nums[idx])+'\n')

f.close()
print("추가 됨.")
print(count)

#file 읽기
f=open("C:/Users/student/PycharmProjects/untitled3/chapter4/file1.txt","r")
lines = f.readlines()
print(lines)
for line in lines:
    print(line,end="") #end는 줄바꿈 없애기.
data = f.read() #하나의 문자열로 받음
print(data)
print(type(data)) #str
print(type(lines)) #list
f.close()

f=open("C:/Users/student/PycharmProjects/untitled3/chapter4/file1.txt","r")
lines = f.readlines()
print(lines)
f.seek(34.,0)
for line in lines:
    print(line,end="") #end는 줄바꿈 없애기.
# f.seek(0,0)
# f.seek(17.0) #시작점을 기준으로 17번째 칸
line = f.readline()
print(line)
print(f.tell())
f.close()

#with 블록을 벗어나는 순간 바로 종료 f.close쓸 필요없음
with open("file1.txt","w") as f:
    f.write("hello python")

#파일의 내용을 리스트로 저장
f = open("file1.txt","r")
f_list = list(f)
print(f_list)
f.close()

#문자열 내에 you라는 값이 몇개 있나 찾아봄.
str= "i love you, you love me"
print(str.count("you"))