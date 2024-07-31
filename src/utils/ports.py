import socket


def is_port_available(port, host='127.0.0.1'):
    """
    Check if a port is available on a given host.

    :param port: Port number to check.
    :param host: Host address to check the port on. Default is '127.0.0.1'.
    :return: True if the port is available, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False


if __name__ == '__main__':
    print(f"12355: {is_port_available(12355)}")
    print(f"12356: {is_port_available(12355)}")
    print(f"12357: {is_port_available(12355)}")
