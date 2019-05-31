PI = 3.141592

#main 반드시 class밖으로 나와야됨.
class Math:
    def solve(self, r):
        return PI * (r **2)
    def sum(self, a,b):
        return a+b


if __name__=="__main__":
    print(PI)
    a = Math()
    print(a.solve(2))
    print(a.sum(PI, 4.4))


#sys라이브러리를 import한다.
import sys
print(sys.path) #python 라이브러리 path - 모듈을 찾는 부분
