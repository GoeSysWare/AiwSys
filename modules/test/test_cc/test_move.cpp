
#include <string>
#include <vector>
#include <iostream>
#include <memory>

class B;
class A
{
public:
    A() { std::cout << "class A : constructor" << std::endl; }
    ~A() { std::cout << "class A : destructor" << std::endl; }
    void referB(std::shared_ptr<B> test_ptr) { _B_Ptr = test_ptr; }
    std::auto_ptr
    // std::shared_ptr<B> _B_Ptr;
    std::weak_ptr<B> _B_Ptr;

};
class B
{
public:
    B() { std::cout << "class B : constructor" << std::endl; }
    ~B() { std::cout << "class B : destructor" << std::endl; }
    void referA(std::shared_ptr<A> test_ptr) { _A_Ptr = test_ptr; }
    // std::shared_ptr<A> _A_Ptr;
    std::weak_ptr<B> _B_Ptr;

};

std::weak_ptr<int> gw;

void f()
{
    if (auto spt = gw.lock())
    { // 使用之前必须复制到 shared_ptr
        std::cout << *spt << "\n";
    }
    else
    {
        std::cout << "gw is expired\n";
    }
}

int main()
{
    std::string str = "Hello";
    std::vector<std::string> v;
    //调用常规的拷贝构造函数，新建字符数组，拷贝数据
    v.push_back(str);
    std::cout << "After copy, str is \"" << str << "\"\n";
    //调用移动构造函数，掏空str，掏空后，最好不要使用str
    v.push_back(std::move(str));
    std::cout << "After move, str is \"" << str << "\"\n";
    std::cout << "The contents of the vector are \"" << v[0] << "\", \"" << v[1] << "\"\n";
    std::shared_ptr<int> p(new int(22334));
    std::cout << "p count: \"" << p.use_count() << "\"\n";
    // std::shared_ptr<int> p1 = std::move(p);
    std::shared_ptr<int> p1 = std::move(p);
    std::cout << "p count: \"" << p.use_count() << "\"\n";
    std::cout << "p1 count: \"" << p1.use_count() << "\"\n";
    std::cout << "shared ==: \"" << (nullptr == p1) << "\"\n";
    std::cout << "shared ==: \"" << (p == nullptr) << "\"\n";
    std::cout << "shared ==: \"" << true << "\"\n";

    // test
    {
        std::shared_ptr<A> ptr_a = std::make_shared<A>(); //A引用计算器为1
        std::shared_ptr<B> ptr_b = std::make_shared<B>(); //B引用计算器为1
        ptr_a->referB(ptr_b);                             // B引用计算器加1
        ptr_b->referA(ptr_a);                             // A引用计算器加1
        std::cout << "A count ==: \"" << ptr_a.use_count() << "\"\n";
        std::cout << "B count ==: \"" << ptr_b.use_count() << "\"\n";
    }

    {
        {
            auto sp = std::make_shared<int>(42);
            gw = sp;
            f();
        }
        f();
    }
    {
        std::unique_ptr<int> up1(new int(11)); // 无法复制的unique_ptr

        // std::shared_ptr<int> up46(std::move(up1)); 

        //unique_ptr<int> up2 = up1;        // err, 不能通过编译
        std::cout << *up1 << std::endl; // 11

        std::unique_ptr<int> up3 = std::move(up1); // 现在p3是数据的唯一的unique_ptr

        std::cout << *up3 << std::endl; // 11
        //std::cout << *up1 << std::endl;   // err, 运行时错误
        up3.reset(); // 显式释放内存
        up1.reset(); // 不会导致运行时错误
        //std::cout << *up3 << std::endl;   // err, 运行时错误

        std::unique_ptr<int> up4(new int(22)); // 无法复制的unique_ptr
        up4.reset(new int(44));                //"绑定"动态对象
        std::cout << *up4 << std::endl;        // 44

        up4 = nullptr; //显式销毁所指对象，同时智能指针变为空指针。与up4.reset()等价

        std::unique_ptr<int> up5(new int(55));
        int *p = up5.release(); //只是释放控制权，不会释放内存
        std::cout << *p << std::endl;
        //cout << *up5 << endl; // err, 运行时错误
        delete p; //释放堆区资源
    }
    return 0;
}
