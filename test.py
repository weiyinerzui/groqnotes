from weasyprint import HTML

HTML(string='<h1>Hello, World!</h1>').write_pdf('test.pdf')
print("执行完成")
