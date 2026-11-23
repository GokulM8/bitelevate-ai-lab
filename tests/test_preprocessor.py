"""Unit tests for clean_ticket_text using messy, real-world style tickets."""
from app.services.preprocessor import clean_ticket_text


# --- Empty input ------------------------------------------------------------------
def test_empty_and_none_return_empty_string():
    assert clean_ticket_text("") == ""
    assert clean_ticket_text(None) == ""


# --- HTML -----------------------------------------------------------------------------
def test_strips_html_tags_and_entities():
    raw = "<div><p>My <b>invoice</b> is wrong&nbsp;&amp; overdue</p></div>"
    assert clean_ticket_text(raw) == "my invoice is wrong & overdue"


def test_strips_escaped_html_tags():
    assert clean_ticket_text("Click &lt;b&gt;Save&lt;/b&gt; fails") == "click save fails"


# --- Email noise ---------------------------------------------------------------------
def test_removes_email_headers():
    raw = (
        "From: John Doe <john@acme.com>\n"
        "Sent: Monday, November 17, 2026\n"
        "To: support@supportpulse.io\n"
        "Subject: Login broken\n"
        "\n"
        "I cannot log into my account."
    )
    assert clean_ticket_text(raw) == "i cannot log into my account."


def test_removes_signature_block():
    raw = "The export button does nothing.\n\nBest regards,\nJohn Doe\nSenior Manager, Acme Corp"
    assert clean_ticket_text(raw) == "the export button does nothing."


def test_removes_mobile_signature_and_quoted_reply():
    raw = (
        "Still broken after the update.\n"
        "On Mon, Nov 17, 2026 at 9:00 AM Support wrote:\n"
        "> Please try clearing your cache.\n"
        "Sent from my iPhone"
    )
    assert clean_ticket_text(raw) == "still broken after the update."


def test_thanks_inside_a_sentence_is_kept():
    raw = "Thanks for the quick reply, but the issue is back."
    assert clean_ticket_text(raw) == "thanks for the quick reply, but the issue is back."


# --- Timestamps ------------------------------------------------------------------------
def test_removes_timestamps():
    raw = "[2026-11-20 14:13:05] Payment failed again at 2:14 AM on 20/11/2026"
    assert clean_ticket_text(raw) == "payment failed again at on"


# --- PII ---------------------------------------------------------------------------------
def test_masks_email_addresses():
    raw = "Please reset the account for jane.doe+billing@example.co.in"
    assert clean_ticket_text(raw) == "please reset the account for [EMAIL]"


def test_masks_phone_numbers():
    raw = "Call me on +91 98765 43210 or (555) 123-4567"
    assert clean_ticket_text(raw) == "call me on [PHONE] or [PHONE]"


def test_short_numbers_are_not_masked():
    raw = "Order #4521 charged 1200 rupees twice"
    assert clean_ticket_text(raw) == "order #4521 charged 1200 rupees twice"


# --- Punctuation & whitespace ----------------------------------------------------------------
def test_collapses_duplicate_punctuation():
    assert clean_ticket_text("WHY is this broken?!?!! Fix it!!!") == "why is this broken? fix it!"


def test_normalizes_whitespace():
    raw = "  App   crashes \n\n\n\t on   startup   .  "
    assert clean_ticket_text(raw) == "app crashes on startup."


# --- Case normalization with preservation ------------------------------------------------------
def test_lowercases_but_preserves_error_identifiers():
    raw = "Getting ERR_404 and HTTP_500 plus a NullPointerException on CHECKOUT"
    assert clean_ticket_text(raw) == (
        "getting ERR_404 and HTTP_500 plus a NullPointerException on checkout"
    )


def test_preserves_inline_code_and_code_blocks():
    raw = (
        "Running `pip install SupportPulse==2.0` FAILS!!\n"
        "```\nTraceback:\n  KeyError: 'API_KEY'\n```"
    )
    assert clean_ticket_text(raw) == (
        "running `pip install SupportPulse==2.0` fails! "
        "```\nTraceback:\n  KeyError: 'API_KEY'\n```"
    )


def test_html_code_tag_is_preserved_as_code():
    raw = "<p>Error in <code>getUserProfile(ID)</code> call</p>"
    assert clean_ticket_text(raw) == "error in `getUserProfile(ID)` call"


# --- End-to-end -----------------------------------------------------------------------------------
def test_full_messy_ticket():
    raw = (
        "From: Priya Sharma <priya.sharma@acme.com>\n"
        "Subject: URGENT!!! Login issue\n"
        "\n"
        "<html><body><p>Hi Team,</p>"
        "<p>Since 2026-11-19 09:42 I keep getting <b>ERR_401</b> when I   log in!!!</p>"
        "<p>Please call me at +91-98765-43210.</p></body></html>\n"
        "\n"
        "Regards,\n"
        "Priya Sharma\n"
        "Acme Corp | priya.sharma@acme.com"
    )
    assert clean_ticket_text(raw) == (
        "hi team, since i keep getting ERR_401 when i log in! please call me at [PHONE]."
    )
