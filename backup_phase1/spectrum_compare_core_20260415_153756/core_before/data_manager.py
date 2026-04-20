import streamlit as st


class SignalRegistry:
    @staticmethod
    def initialize():
        if "signals" not in st.session_state:
            st.session_state.signals = {}

    @staticmethod
    def add_signal(signal):
        st.session_state.signals[signal.file_name] = signal

    @staticmethod
    def get_signal(file_name):
        return st.session_state.signals.get(file_name)

    @staticmethod
    def get_all_signals():
        return st.session_state.signals

    @staticmethod
    def remove_signal(file_name):
        if file_name in st.session_state.signals:
            del st.session_state.signals[file_name]

    @staticmethod
    def clear():
        st.session_state.signals = {}