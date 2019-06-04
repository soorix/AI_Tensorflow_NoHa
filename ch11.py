# 4/0
try:
    4/0 #예외가 발생할 가능성이 있는 코드
except ZeroDivisionError as e:
    print(e) #예외가 발생했을 때의 코드



while True:
    num1 = input("\n분자: ")
    if num1 == 'q':
        break
    num2 = input("\n분모: ")
    try:
        result = int(num1)/int(num2)
        print(result)
    except ZeroDivisionError:
        print("분모에 0이 올 수 없습니다.")
print("종료")


def exception_test():
    print("start")
    try:
        print(2+'2')
    except TypeError as e:
        print("TypeError : {0}".format(e))
    print('end')

exception_test()

import traceback
def exception_test():
    print("start")
    try:
        print(2+'2')
    except TypeError as e:
        #좀더 자세하게 traceback정보를 보고싶음.
        traceback.print_exc()
    print('end')

exception_test()


#######################################12강 정규표현식
import re #정규표현식 모듈
str = "My id number is kim1005"
a = re.findall("i",str)
b = re.findall("My",str)
c = re.findall('[a-z]',str) #소문자를 모두 찾기
c = re.findall('[^a-z]',str) #소문자가 아닌 것들
d = re.findall("[A-Z]",str) #대문자
print(d)
e = re.findall("[a-zA-Z]",str)
print(e)
f = re.findall("[\w]",str) #영문자, 숫자, _ 찾기
print(f)
print(c)
print(a)
print(b)

def pwd_check(pwd):
    if len(pwd) < 6 or len(pwd) > 12:
        print(pwd, "의 길이가 적당하지 않습니다.")
        return False

    if re.findall("[a-zA-Z0-9]", pwd)[0] != pwd:
        print(pwd," ==> 숫자와 영문자로만 구성되어야 합니다.")
        return False

    if len(re.findall("[a-z]", pwd))==0 or len(re.findall("[A-Z]",pwd))==0:
        print(pwd,"==> 대문자와 소문자가 모두 필요합니다.")
        return False

    print(pwd, "==>올바름 비밀 번호입니다.")
    return True


pwd_check('123')
pwd_check('1234ssdgD#')
pwd_check('123Abc')

#^[]시작, [^] not, $끝 \.마침표

def email_check(email):
    exp = re.findall("^[a-z0-9]{2,}@[a-z0-9]{2,}\.[a-z]{2.}$",email)
    if len(exp) == 0:
        print(email, "==>잘못된 메일 형식")
        return
    print(email, "==> 올바른 메일 주소")
    return

email_check("Kim@gamil")
email_check("Kim@gmail.com")
email_check("Kim")