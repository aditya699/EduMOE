1.python -m venv venv

2.source venv/bin/activate

3.nohup python -u train.py > training.log 2>&1 &

4.tail -f training.log

5.pkill -f train.py

6.nvidia-smi

7.deactivate

8.ps aux | grep python
