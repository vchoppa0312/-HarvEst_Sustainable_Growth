# Imports
import streamlit as st
from web3 import Web3
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
import os
import requests
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


def main():
    st.title('''Welcome to the Discovery Page :earth_asia:''')
    #st.write('A Sneak Peak into the ***Blockchain*** & ***Tokenomics***')

    #col1,col2 = st.columns([4,1],gap='small')

    with st.form(key='Token'):
            st.write('#### A Sneak Peak into the ***Blockchain*** & ***Tokenomics***')
            generate = st.form_submit_button("Generate")
            if generate:
                mnemo = Mnemonic("english")
                words = mnemo.generate(strength=128)
                st.markdown("""##### *Generate Your Digital Wallet and Ethereum Account*""")
                st.write('Secure the Phrase carefully for safe access and security of you Digital Wallet!')
                st.write(f"#### {words}")
                st.write('And No! You Cannot Recover this phrase again')


    
                account,private = generate_account(words)
                st.markdown("## Wallet Address:")
                st.write(f"{account.address}")


                # st.text("\n")
                # st.markdown("## An Ethereum Transaction:")

                #receiver = st.text_input('Select Receiver account')
                ether = 10

# @TODO
# Create a button that calls the `send_transaction` function and returns the transaction hash
    if st.button('Harvest Tokens'):
                    st.text("\n")
                    st.markdown("## Wallet Balance:")

                    ether_balance=get_balance
                    st.write(f"{ether_balance}")
                    st.markdown('## Visit again to Harvest More')






    st.header('COME SOON FOR MORE DEVELOPMENTS & INSIGHTS!')

    st.subheader('Tell Us What You Think')

    contact_form = ''' <form action="https://formsubmit.co/joannelaomoc@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder ="Your Name" required>
     <input type="email" name="email" placeholder = "Your Email"required>
     <textarea name="message" placeholder = "Your Message Here" require></textarea>
     <button type="submit">Send</button>
     </form>
'''
    left_column, right_column = st.columns(2)
    with left_column:
          st.markdown(contact_form, unsafe_allow_html = True)
    with right_column:
          st.empty()

    



if st.button("Home Page"):
    st.switch_page("pages/2_Investment_Info.py")

if __name__ == "__main__":
    main()