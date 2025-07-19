def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return bytestring.decode("utf-8")
    # return "".join([bytes([b]).decode("utf-8") for b in bytestring])

print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))
print("hello".encode("utf-8"))