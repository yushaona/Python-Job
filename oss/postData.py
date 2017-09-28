import _md5 as md5
import hashlib
import base64
import hmac
from optparse import OptionParser
def convert_base64(input):
    s= bytes(input,encoding="utf8")
    return str(base64.b64encode(s),encoding="utf8")

def get_sign_policy(key, policy):
    return str(base64.b64encode(hmac.new(bytes(key,encoding="utf8"), bytes(policy,encoding="utf8"), hashlib.sha1).digest()),encoding="utf8")

def get_form(bucket, endpoint, access_key_id, access_key_secret, out):
    #1 构建一个Post Policy
    policy="{\"expiration\":\"2115-01-27T10:56:19Z\",\"conditions\":[[\"content-length-range\", 0, 5368709120]]}"
    print("policy: %s" % policy)
    #2 将Policy字符串进行base64编码
    base64policy = convert_base64(policy)
    print("base64_encode_policy: %s" % base64policy)
    #3 用OSS的AccessKeySecret对编码后的Policy进行签名
    signature = get_sign_policy(access_key_secret, base64policy)
    #4 构建上传的HTML页面
    form = '''
    <html>
        <meta http-equiv=content-type content="text/html; charset=UTF-8">
        <head><title>OSS表单上传(PostObject)</title></head>
        <body>
            <form  action="http://%s.%s" method="post" enctype="multipart/form-data">
                <input type="text" name="OSSAccessKeyId" value="%s">
                <input type="text" name="policy" value="%s">
                <input type="text" name="Signature" value="%s">
                <input type="text" name="key" value="upload/${filename}">
                <input type="text" name="success_action_redirect" value="http://oss.aliyun.com">
                <input type="text" name="success_action_status" value="201">
                <input name="file" type="file" id="file">
                <input name="submit" value="Upload" type="submit">
            </form>
        </body>
    </html>
    ''' % (bucket, endpoint, access_key_id, base64policy, signature)
    f = open(out, "wb")
    f.write(bytes(form,encoding="utf8"))
    f.close()
    print("form is saved into %s" % out)
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--bucket", dest="bucket",default="dtcollage", help="specify")
    parser.add_option("", "--endpoint", dest="endpoint",default="oss-cn-qingdao.aliyuncs.com", help="specify")
    parser.add_option("", "--id", dest="id",default="LTAIXUxozaqB17nz", help="access_key_id")
    parser.add_option("", "--key", dest="key", default="Qrlqi1rxBzlII4VQ6xSiogkO7UXzI7",help="access_key_secret")
    parser.add_option("", "--out", dest="out",default="post.html", help="out put form")
    (opts, args) = parser.parse_args()
    if opts.bucket and opts.endpoint and opts.id and opts.key and opts.out:
        get_form(opts.bucket, opts.endpoint, opts.id, opts.key, opts.out)
    else:
        print ("python %s --bucket=your-bucket --endpoint=oss-cn-hangzhou.aliyuncs.com --id=your-access-key-id --key=your-access-key-secret --out=out-put-form-name" % __file__)