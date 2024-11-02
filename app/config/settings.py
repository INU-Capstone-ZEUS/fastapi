import socket

# 현재 컴퓨터의 로컬 IP 주소 추출 함수
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP용 소켓 생성
    try:
        s.connect(("8.8.8.8", 80)) # Google의 DNS 서버 (8.8.8.8)로 연결 시도
        ip = s.getsockname()[0] # 소켓을 통해 얻은 로컬 IP 주소
    except Exception:
        ip = "127.0.0.1" # 예외 발생 시 기본값으로 'localhost' IP를 사용
    finally:
        s.close() # 소켓을 닫아 리소스 해제
    return ip

IP_NUM = get_local_ip()
PORT_NUM = "8080"