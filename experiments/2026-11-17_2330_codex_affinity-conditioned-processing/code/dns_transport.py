"""Process-local HTTPS DNS fallback; no system configuration or package changes."""

import runpy
import ipaddress
import socket
import sys
from pathlib import Path

import dns.resolver
import dns.exception


def make_getaddrinfo(resolver, original):
    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if host is None or host == "localhost" or flags & socket.AI_NUMERICHOST:
            return original(host, port, family, type, proto, flags)
        name = host.decode("ascii") if isinstance(host, bytes) else host
        try:
            ipaddress.ip_address(name)
        except ValueError:
            pass
        else:
            return original(host, port, family, type, proto, flags)
        try:
            answers = resolver.resolve_name(name, family=family)
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer) as error:
            raise socket.gaierror(socket.EAI_NONAME, str(error)) from error
        except dns.exception.DNSException as error:
            raise socket.gaierror(socket.EAI_AGAIN, str(error)) from error
        results = []
        for address, address_family in answers.addresses_and_families():
            try:
                numeric = original(address, port, address_family, type, proto,
                                   flags | socket.AI_NUMERICHOST)
            except socket.gaierror:
                # Preserve the OS's address-family availability checks.
                continue
            if flags & socket.AI_CANONNAME:
                cname = answers.canonical_name().to_text(True)
                numeric = [(af, st, pr, cname, addr) for af, st, pr, _, addr in numeric]
            results.extend(numeric)
        if not results:
            raise socket.gaierror(socket.EAI_NONAME, "No address supported by this host")
        return results
    return getaddrinfo


def main():
    arguments = sys.argv[1:]
    module = bool(arguments and arguments[0] == "--module")
    if module:
        arguments = arguments[1:]
    if not arguments:
        raise SystemExit("usage: dns_transport.py [--module] script_or_module [args...]")
    resolver = dns.resolver.Resolver(configure=False)
    resolver.nameservers = ["https://1.1.1.1/dns-query"]
    resolver.lifetime = 15
    resolver.cache = dns.resolver.Cache()
    original = socket.getaddrinfo
    socket.getaddrinfo = make_getaddrinfo(resolver, original)
    sys.argv = arguments
    try:
        if module:
            runpy.run_module(arguments[0], run_name="__main__")
        else:
            sys.path.insert(0, str(Path(arguments[0]).resolve().parent))
            runpy.run_path(arguments[0], run_name="__main__")
    finally:
        socket.getaddrinfo = original


if __name__ == "__main__":
    main()
