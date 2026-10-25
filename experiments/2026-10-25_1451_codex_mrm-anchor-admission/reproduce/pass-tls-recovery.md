# Explicit TLS recovery for public PASS retrieval

The first transfer attempt received a421 response requiring TLS and downloaded
zero bytes. Preserve that terminal receipt. Use ftplib.FTP_TLS with the default
verified SSL context, explicit TLS login and protected data connection (PROT P).
Keep the same published host, source-derived public login, single filename,
limits and password omission. Do not disable certificate verification or weaken
ciphers. Give the recovery its own script/configuration/runtime/transfer receipt.
