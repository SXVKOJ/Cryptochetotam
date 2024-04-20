from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import transpile
from qiskit.circuit.library import QFT
from fractions import Fraction
from math import gcd, pi
import random

from qiskit_ibm_provider import IBMProvider

provider = IBMProvider(token='')

# --- Р¤СѓРЅРєС†РёРё РґР»СЏ RSA ---
def generate_keypair(n_limit=15):
    """Р“РµРЅРµСЂРёСЂСѓРµС‚ РїР°СЂСѓ РєР»СЋС‡РµР№ RSA."""
    while True:
        p = random.randrange(3, n_limit, 2)
        if is_prime(p):
            break
    while True:
        q = random.randrange(3, n_limit, 2)
        if is_prime(q) and p != q:
            break
    N = p * q
    phi = (p - 1) * (q - 1)
    while True:
        e = random.randrange(2, phi)
        if gcd(e, phi) == 1:
            break
    d = mod_inverse(e, phi)
    return (N, e), (N, d)

def is_prime(n):
    """РџСЂРѕРІРµСЂСЏРµС‚, СЏРІР»СЏРµС‚СЃСЏ Р»Рё С‡РёСЃР»Рѕ РїСЂРѕСЃС‚С‹Рј."""
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def mod_inverse(a, m):
    """РќР°С…РѕРґРёС‚ РјРѕРґСѓР»СЊРЅРѕРµ РѕР±СЂР°С‚РЅРѕРµ С‡РёСЃР»Рѕ a РїРѕ РјРѕРґСѓР»СЋ m."""
    m0 = m
    y = 0
    x = 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        t = m
        m = a % m
        a = t
        t = y
        y = x - q * y
        x = t
    if x < 0:
        x = x + m0
    return x

def encrypt(message, public_key):
    """РЁРёС„СЂСѓРµС‚ СЃРѕРѕР±С‰РµРЅРёРµ СЃ РїРѕРјРѕС‰СЊСЋ RSA."""
    N, e = public_key
    encrypted_blocks = []
    for num in message:
        char = chr(num)
        block = ord(char) - ord('a') 
        encrypted_block = pow(block, e, N)
        encrypted_blocks.append(encrypted_block)
    return encrypted_blocks

def decrypt(encrypted_blocks, private_key):
    """Р Р°СЃС€РёС„СЂРѕРІС‹РІР°РµС‚ Р±Р»РѕРєРё СЃ РїРѕРјРѕС‰СЊСЋ RSA."""
    N, d = private_key
    message = []
    for block in encrypted_blocks:
        decrypted_block = pow(block, d, N)
        char = chr(decrypted_block + ord('a')) 
        message.append(char)
    return message

# --- Р¤СѓРЅРєС†РёРё РґР»СЏ Р°Р»РіРѕСЂРёС‚РјР° РЁРѕСЂР° ---
def create_qft(n, inverse=False):
    """РЎРѕР·РґР°РµС‚ РєРІР°РЅС‚РѕРІСѓСЋ СЃС…РµРјСѓ РґР»СЏ QFT РёР»Рё РѕР±СЂР°С‚РЅРѕРіРѕ QFT."""
    qc = QuantumCircuit(n)
    if inverse:
        for qubit in range(n // 2):
            qc.swap(qubit, n - qubit - 1)
        for j in range(n):
            for m in range(j):
                qc.cp(-pi / (2 ** (j - m)), m, j)
            qc.h(j)
    else:
        for j in range(n):
            qc.h(j)
            for m in range(j + 1, n):
                qc.cp(pi / (2 ** (m - j)), j, m)
        for qubit in range(n // 2):
            qc.swap(qubit, n - qubit - 1)
    return qc

def create_adder_mod(n, bigN):
    """РЎРѕР·РґР°РµС‚ РєРІР°РЅС‚РѕРІСѓСЋ СЃС…РµРјСѓ РґР»СЏ СЃР»РѕР¶РµРЅРёСЏ РїРѕ РјРѕРґСѓР»СЋ N."""
    a = QuantumRegister(n, 'a')
    b = QuantumRegister(n + 1, 'b')
    c = QuantumRegister(n, 'c')
    bN = QuantumRegister(n, 'bN')
    t = QuantumRegister(1, 't')
    qc = QuantumCircuit(a, b, c, bN, t) 

    # Р’С‹С‡РёСЃР»РµРЅРёРµ a + b
    for i in range(n - 1):
        add_bits(qc, a[i], b[i], c[i], c[i + 1], c[i + 1], t[0])
    add_bits(qc, a[n - 1], b[n - 1], c[n - 1], b[n], b[n], t[0]) 

    # Р’С‹С‡РёС‚Р°РЅРёРµ N if a + b >= N 
    for i in range(n):
        qc.cx(bN[i], a[i])
    for i in range(n - 1):
        subtract_bits(qc, a[i], b[i], c[i], c[i + 1], c[i + 1], t[0])
    subtract_bits(qc, a[n - 1], b[n - 1], c[n - 1], b[n], b[n], t[0])

    # РЈСЃР»РѕРІРЅРѕРµ РёСЃРїСЂР°РІР»РµРЅРёРµ
    qc.x(b[n])
    qc.cx(b[n], t[0])
    qc.x(b[n])
    tempN = bigN
    i = 0
    while tempN != 0:
        if tempN % 2 != 0:
            for j in range(n):
                qc.cx(t[0], a[j])
        i += 1
        tempN //= 2

    # Р”РѕР±Р°РІР»РµРЅРёРµ N if a + b < N
    for i in range(n - 1):
        add_bits(qc, a[i], b[i], c[i], c[i + 1], c[i + 1], t[0])
    add_bits(qc, a[n - 1], b[n - 1], c[n - 1], b[n], b[n], t[0])
    tempN = bigN
    i = 0
    while tempN != 0:
        if tempN % 2 != 0:
            for j in range(n):
                qc.cx(t[0], a[j])
        i += 1
        tempN //= 2

    # Р’РѕСЃСЃС‚Р°РЅРѕРІР»РµРЅРёРµ a
    for i in range(n):
        qc.cx(bN[i], a[i])

    # Р’С‹С‡РёС‚Р°РЅРёРµ b
    for i in range(n - 1):
        subtract_bits(qc, a[i], b[i], c[i], c[i + 1], c[i + 1], t[0])
    subtract_bits(qc, a[n - 1], b[n - 1], c[n - 1], b[n], b[n], t[0])

    # РЈСЃР»РѕРІРЅРѕРµ РёСЃРїСЂР°РІР»РµРЅРёРµ
    qc.cx(b[n], t[0])
    for i in range(n - 1):
        add_bits(qc, a[i], b[i], c[i], c[i + 1], c[i + 1], t[0])
    add_bits(qc, a[n - 1], b[n - 1], c[n - 1], b[n], b[n], t[0])

    return qc

def add_bits(qc, a, b, c, carry_in, carry_out, temp_qubit): 
    qc.ccx(carry_in, a, b)
    qc.cx(a, b)
    qc.ccx(carry_in, b, temp_qubit) 
    qc.cx(temp_qubit, carry_out) 

def subtract_bits(qc, a, b, c, carry_in, carry_out, temp_qubit): 
    qc.ccx(carry_in, b, temp_qubit) 
    qc.cx(a, b)
    qc.ccx(carry_in, a, b)
    qc.cx(temp_qubit, carry_out)

def create_ctrl_mult_mod(n, bigN, a):
    """РЎРѕР·РґР°РµС‚ РєРІР°РЅС‚РѕРІСѓСЋ СЃС…РµРјСѓ РґР»СЏ СѓРїСЂР°РІР»СЏРµРјРѕРіРѕ СѓРјРЅРѕР¶РµРЅРёСЏ РїРѕ РјРѕРґСѓР»СЋ N."""
    x = QuantumRegister(1, 'x')
    z = QuantumRegister(n, 'z')
    qc = QuantumCircuit(x, z)
    adder_mod_qc = create_adder_mod(n, bigN)
    adder_qubits = adder_mod_qc.qubits 
    for reg in adder_mod_qc.qregs:
        qc.add_register(reg)
    for i in range(n):
        control_qubits = [x[0]] + adder_qubits 
        qc.append(adder_mod_qc.control(1), control_qubits)
    return qc

"""def create_mod_exp(n, bigN, y, nx):
    РЎРѕР·РґР°РµС‚ РєРІР°РЅС‚РѕРІСѓСЋ СЃС…РµРјСѓ РґР»СЏ РјРѕРґСѓР»СЊРЅРѕРіРѕ РІРѕР·РІРµРґРµРЅРёСЏ РІ СЃС‚РµРїРµРЅСЊ (a^x mod N).
    x = QuantumRegister(nx, 'x')
    z = QuantumRegister(n, 'z')
    qc = QuantumCircuit(x, z)
    qubit_map = {}
    m = y
    for i in range(nx):
        ctrl_mult_mod_qc = create_ctrl_mult_mod(n, bigN, m)
        all_qubits = ctrl_mult_mod_qc.qubits
        # РЎРѕРїРѕСЃС‚Р°РІР»СЏРµРј РєСѓР±РёС‚С‹ РёР· x РІ ctrl_mult_mod_qc СЃ РєСѓР±РёС‚Р°РјРё x РІ qc
        qubit_map.update({qb: x[i] for qb in ctrl_mult_mod_qc.qregs[0]})
        print(qubit_map)
        # Р”РѕР±Р°РІР»СЏРµРј РІСЃРµ СЂРµРіРёСЃС‚СЂС‹ РёР· ctrl_mult_mod_qc, РєСЂРѕРјРµ x, РІ qc
        for reg in ctrl_mult_mod_qc.qregs[1:]:
            if reg.name != 'z':
                qc.add_register(reg)
        # Р”РѕР±Р°РІР»СЏРµРј СѓРїСЂР°РІР»СЏСЋС‰СѓСЋ СЃС…РµРјСѓ СЃ СЃРѕРїРѕСЃС‚Р°РІР»РµРЅРЅС‹РјРё РєСѓР±РёС‚Р°РјРё
        qc.append(ctrl_mult_mod_qc, all_qubits, qubit_map)
        m = (m * m) % bigN
    return qc
"""

def create_ctrl_mult_mod(n, bigN, m): 
    x = QuantumRegister(1, 'x') 
    z = QuantumRegister(n, 'z') 
    a = QuantumRegister(n, 'a') 
    b = QuantumRegister(n+1, 'b') 
    c = QuantumRegister(n, 'c') 
    bN = QuantumRegister(n, 'bN') 
    t = QuantumRegister(1, 't') 
    qc = QuantumCircuit(x, z, a, b, c, bN, t) 

    next_mod = m # m 2^0 mod N 
    for j in range(n): 
        temp_mod = next_mod 
        i = 0 

        while temp_mod != 0: 
            if temp_mod % 2 != 0: 
                qc.ccx(x[0], z[j], a[i]) 

            i = i + 1
            temp_mod = temp_mod // 2 

        qc.append(create_adder_mod(n, bigN), list(a) + list(b) + list(c) + list(bN) + list(t)) 
        temp_mod = next_mod 

        i = 0 
        while temp_mod != 0: 
            if temp_mod % 2 != 0: 
                qc.ccx(x[0], z[j], a[i]) 

            i = i + 1 
            temp_mod = temp_mod // 2 

        next_mod = (next_mod * 2) % bigN  
        
    return qc.to_gate(label='ctrl_mult_mod') 


def create_ctrl_mult_mod_inv(n, bigN, m): 
    qc = QuantumCircuit(5*n+3) 
    qc.append(create_ctrl_mult_mod(n, bigN, m).inverse(), range(5*n+3)) 
    return qc.to_gate(label='ctrl_mult_mod_inv') 


def create_mod_exp(n, bigN, y, nx): 
    x = QuantumRegister(nx, 'x') 
    z = QuantumRegister(n, 'z') 
    a = QuantumRegister(n, 'a') 
    b = QuantumRegister(n+1, 'b') 
    c = QuantumRegister(n, 'c') 
    bN = QuantumRegister(n, 'bN') 
    t = QuantumRegister(1, 't') 
    qc = QuantumCircuit(x, z, a, b, c, bN, t) 
    m = y 

    for i in range(nx): 
        qc.append(create_ctrl_mult_mod(n, bigN, m),  
            [x[i]] + z[:] + a[:] + b[:] + c[:] + bN[:] + list(t)) 
        for j in range(n): 
            qc.cswap(x[i], z[j], b[j]) 

        m_inv = mod_inverse(m, bigN) 
        qc.append(create_ctrl_mult_mod_inv(n, bigN, m_inv),  
            [x[i]] + z[:] + a[:] + b[:] + c[:] + bN[:] + list(t)) 
        m = (m * m) % bigN 
        
    return qc.to_gate(label='mod_exp') 

def shors_algorithm(N):
    """Р РµР°Р»РёР·СѓРµС‚ Р°Р»РіРѕСЂРёС‚Рј РЁРѕСЂР° РґР»СЏ С„Р°РєС‚РѕСЂРёР·Р°С†РёРё N."""
    n_count = N.bit_length()
    a = random.randint(2, N - 1)
    # РџСЂРѕРІРµСЂРєР° РЅР° РЅР°Р»РёС‡РёРµ РѕР±С‰РµРіРѕ РґРµР»РёС‚РµР»СЏ
    if gcd(a, N) != 1:
        return gcd(a, N)
    # РљРІР°РЅС‚РѕРІР°СЏ С‡Р°СЃС‚СЊ (РїРѕРёСЃРє РїРµСЂРёРѕРґР°)
    x = QuantumRegister(2 * n_count)
    c = ClassicalRegister(2 * n_count)
    qc = QuantumCircuit(x, c)
    qc.h(x[:n_count])  # РЎСѓРїРµСЂРїРѕР·РёС†РёСЏ
    qc.x(x[n_count])  # РЈСЃС‚Р°РЅРѕРІРєР° z=1
    # РњРѕРґСѓР»СЊРЅРѕРµ РІРѕР·РІРµРґРµРЅРёРµ РІ СЃС‚РµРїРµРЅСЊ (a^x mod N)
    qc.append(create_mod_exp(n_count, N, a, n_count), x[:] + c[:]) 
    # РћР±СЂР°С‚РЅРѕРµ РїСЂРµРѕР±СЂР°Р·РѕРІР°РЅРёРµ Р¤СѓСЂСЊРµ
    qc.append(create_qft(n_count, inverse=True), x[:n_count])
    # РР·РјРµСЂРµРЅРёРµ
    qc.measure(x[:n_count], c[:n_count])
    # РѕС‚РїСЂР°РІР»СЏРµРј РєРІР°РЅС‚РѕРІСѓСЋ СЃС…РµРјСѓ РЅР° СЃРµСЂРІРµСЂ
    backend = provider.get_backend("ibmq_qasm_simulator")
    job = backend.run(transpile(qc, backend), shots=1024)
    counts = job.result().get_counts()
    # РљР»Р°СЃСЃРёС‡РµСЃРєР°СЏ РїРѕСЃС‚РѕР±СЂР°Р±РѕС‚РєР°
    for output in counts:
        phase = int(output, 2) / 2**n_count
        frac = Fraction(phase).limit_denominator(N)
        period = frac.denominator
        if period % 2 == 0 and pow(a, period // 2, N) != N - 1:
            return gcd(pow(a, period // 2) + 1, N)
    return None


def extended_gcd(num1, num2):
    if num1 == 0:
        return (num2, 0, 1)
    else:
        div, x, y = extended_gcd(num2 % num1, num1)
    return (div, y - (num2 // num1) * x, x)


def find_d(e, phi):
  """
  РќР°С…РѕРґРёС‚ РјРѕРґСѓР»СЊРЅРѕРµ РѕР±СЂР°С‚РЅРѕРµ d С‡РёСЃР»Р° e РїРѕ РјРѕРґСѓР»СЋ phi.

  Args:
      e: РћС‚РєСЂС‹С‚Р°СЏ СЌРєСЃРїРѕРЅРµРЅС‚Р°.
      phi: Р¤СѓРЅРєС†РёСЏ Р­Р№Р»РµСЂР° РѕС‚ N (phi(N)).

  Returns:
      РњРѕРґСѓР»СЊРЅРѕРµ РѕР±СЂР°С‚РЅРѕРµ d, С‚Р°РєРѕРµ С‡С‚Рѕ (d * e) % phi = 1.
  """
  gcd, x, _ = extended_gcd(e, phi)
  if gcd != 1:
    raise ValueError("e Рё phi(N) РґРѕР»Р¶РЅС‹ Р±С‹С‚СЊ РІР·Р°РёРјРЅРѕ РїСЂРѕСЃС‚С‹РјРё")
  else:
    d = x % phi
    return d

def rsa_private_key(p, q, e):
  """
  Р’С‹С‡РёСЃР»СЏРµС‚ Р·Р°РєСЂС‹С‚С‹Р№ РєР»СЋС‡ RSA РїРѕ Р·Р°РґР°РЅРЅС‹Рј p, q Рё e.

  Args:
      p: РџРµСЂРІС‹Р№ РїСЂРѕСЃС‚РѕР№ РјРЅРѕР¶РёС‚РµР»СЊ С‡РёСЃР»Р° N.
      q: Р’С‚РѕСЂРѕР№ РїСЂРѕСЃС‚РѕР№ РјРЅРѕР¶РёС‚РµР»СЊ С‡РёСЃР»Р° N.
      e: РћС‚РєСЂС‹С‚Р°СЏ СЌРєСЃРїРѕРЅРµРЅС‚Р°.

  Returns:
      Р—Р°РєСЂС‹С‚С‹Р№ РєР»СЋС‡ RSA (d).
  """
  n = p * q
  phi = (p - 1) * (q - 1)
  d = find_d(e, phi)
  return d


# --- РџСЂРёРјРµСЂ РёСЃРїРѕР»СЊР·РѕРІР°РЅРёСЏ ---
msg_text = "hello"
message = [ord(msg_text[i]) for i in range(len(msg_text))] 
public_key, _ = generate_keypair() # С„СѓРЅРєС†РёСЏ РІРѕР·РІСЂР°С‰Р°РµС‚ РїСѓР±Р»РёС‡РЅС‹Р№ Рё РїСЂРёРІР°С‚РЅС‹Р№ РєР»СЋС‡ (РІС‚РѕСЂРѕР№ РјС‹ 'Р·Р°С‚С‹РєР°РµРј')
print("РћС‚РєСЂС‹С‚С‹Р№ РєР»СЋС‡ (N, e):", public_key)
encrypted_blocks = encrypt(message, public_key)
print("РСЃС…РѕРґРЅРѕРµ СЃРѕРѕР±С‰РµРЅРёРµ:", msg_text)
print("Р—Р°С€РёС„СЂРѕРІР°РЅРЅС‹Рµ Р±Р»РѕРєРё:", encrypted_blocks, "РІРёРґ РІ ASCII:", f"({''.join([chr(encrypted_blocks[i]) for i in range(len(message))])})")
N = public_key[0]
p = shors_algorithm(N)
q = N // p
print("РќР°Р№РґРµРЅРЅС‹Рµ РјРЅРѕР¶РёС‚РµР»Рё:", p, q)
# РЅРµ РёРјРµСЏ РїСЂРёРІР°С‚РЅРѕРіРѕ РєР»СЋС‡Р°, Р·РЅР°СЏ РЅР°Р№РґРµРЅРЅС‹Рµ РјРЅРѕР¶РёС‚РµР»Рё p Рё q РѕС‚ С‡РёСЃР»Р° N, РЅР°С…РѕРґРёРј Р·РЅР°С‡РµРЅРёРµ РїСЂРёРІР°С‚РЅРѕРіРѕ РєР»СЋС‡Р° d
private_key = N, rsa_private_key(p, q, public_key[1])
decrypted_message = decrypt(encrypted_blocks, private_key)
print("Р Р°СЃС€РёС„СЂРѕРІР°РЅРЅРѕРµ СЃРѕРѕР±С‰РµРЅРёРµ:", ''.join(decrypted_message), [ord(decrypted_message[i]) for i in range(len(decrypted_message))])
