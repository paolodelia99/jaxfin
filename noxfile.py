# noxfile.py
import nox


@nox.session(venv_backend='venv',
             reuse_venv=True)
def tests(session):
    session.run("pytest", "--junit-xml=test_report.xml")
