Assume your workspace is M:, and you need create M:\Registry for storing parameters.

Here we give the minimal required parameters. 

### devices

located in `Registry\Servers\devices`

- labone_ip
- microwave_server
- microwave_source
- ziHD_id
- ziQA_id



Example of python dictionary

```python
{'labone_ip': '10.00.0.000',
 'microwave_server': 'anritsu_server',
 'microwave_source': [('anritsu_r_1',
   'dirac GPIB Bus - TCPIP0::192.168.1.240::inst0::INSTR'),
  ('anritsu_xy_1', 'dirac GPIB Bus - TCPIP0::192.168.1.241::inst0::INSTR')],
 'ziHD_id': [('hd_1', 'dev8334')],
 'ziQA_id': [('qa_1', 'dev2591')]}
```

