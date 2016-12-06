from IPython import parallel
from IPython import embed
import socket
def temp_func(k):
        import socket
        return socket.gethostname()

client_list = parallel.Client(profile="default")
print("Running on:",len(client_list.ids))
print(client_list.ids)
view = client_list.load_balanced_view()
r = view.map(temp_func,[i for i in range(len(client_list.ids))])
r.get()
for p in r:
        print(p)