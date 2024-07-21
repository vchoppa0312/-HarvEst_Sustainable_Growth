# Imports
import streamlit as st
from web3 import Web3
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
import os
import requests
from dotenv import load_dotenv
load_dotenv()
from bip44 import Wallet
from web3 import Account
from web3 import middleware
from web3.gas_strategies.time_based import medium_gas_price_strategy
from web3 import Web3
from mnemonic import Mnemonic

def generate_account(words):
    """Generating Your Digital Wallet and Ethereum account from displayed seed phrase."""
    mnemonic = words
    wallet = Wallet(mnemonic)
    private, public = wallet.derive_account("eth")
    account = Account.from_key(private)
    return account, private

def get_balance(w3,address):
    wei_balance = w3.eth.get_balance(address)
    ether = w3.from_wei(wei_balance,  'ether')
    return ether
    
def send_transaction(w3,account, receiver, ether, private):
    w3.eth.set_gas_price_strategy(medium_gas_price_strategy)
    wei_value = w3.to_wei(ether, 'ether')
    gas_estimate = w3.eth.estimate_gas ({ 
        "to": receiver, 
        "from": account.address, 
        "value" : wei_value
        })
    raw_tx = {
        'to': receiver,
        'from': account.address,
        'value': wei_value,
        'nonce' : w3.eth.get_transaction_count(account.address),
        'gas': gas_estimate,
        'gasPrice': w3.eth.generate_gas_price()
    }

    signed_tx = w3.eth.account.sign_transaction(raw_tx, private)
    
    return w3.eth.send_raw_transaction(signed_tx.rawTransaction)