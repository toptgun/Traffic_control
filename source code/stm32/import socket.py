import socket


def connect(a):
    data = (f'{a}')
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('10.10.141.22', 5001))
    server_socket.listen(15)
    client_socket, addr = server_socket.accept()
    client_socket.send(data.encode('utf-8'))
    client_socket.close()
    server_socket.close()
    
def main():
    while 1:
        connect(10)
if __name__ == "__main__":
    main()