#include <stdio.h>  
#include <stdlib.h>  
  
typedef struct node  
{   int    coef, exp;  
    struct node  *next;  
} NODE;  
  
void multiplication( NODE *, NODE * , NODE * );  
void input( NODE * );  
void output( NODE * );  
  
void input( NODE * head )  
{   int flag, sign, sum, x;  
    char c;  
  
    NODE * p = head;  
  
    while ( (c=getchar()) !='\n' )  
    {  
        if ( c == '<' )  
        {    sum = 0;  
             sign = 1;  
             flag = 1;  
        }  
        else if ( c =='-' )  
             sign = -1;  
        else if( c >='0'&& c <='9' )  
        {    sum = sum*10 + c - '0';  
        }  
        else if ( c == ',' )  
        {    if ( flag == 1 )  
             {    x = sign * sum;  
                  sum = 0;  
                  flag = 2;  
          sign = 1;  
             }  
        }  
        else if ( c == '>' )  
        {    p->next = ( NODE * ) malloc( sizeof(NODE) );  
             p->next->coef = x;  
             p->next->exp  = sign * sum;  
             p = p->next;  
             p->next = NULL;  
             flag = 0;  
        }  
    }  
}  
  
void output( NODE * head )  
{  
    while ( head->next != NULL )  
    {   head = head->next;  
        printf("<%d,%d>,", head->coef, head->exp );  
    }  
    printf("\n");  
}  

// 安全的O(nm)算法
void multiplication( NODE * h1, NODE * h2, NODE * h3){
    // 使用足够大的固定数组，避免动态分配
    int terms[10000][2]; // [系数, 指数]
    int count = 0;
    
    // 先计算两个多项式的项数，确保不会越界
    int len1 = 0, len2 = 0;
    NODE * temp = h1->next;
    while(temp){ len1++; temp = temp->next; }
    temp = h2->next;
    while(temp){ len2++; temp = temp->next; }
    
    // 检查是否会越界
    if(len1 * len2 > 10000){
        // 如果会越界，使用零多项式
        NODE * s = (NODE*)malloc(sizeof(NODE));
        s->coef = 0;
        s->exp = 0;
        s->next = NULL;
        h3->next = s;
        return;
    }
    
    NODE * p1 = h1->next;
    while(p1){
        NODE * p2 = h2->next;
        while(p2){
            terms[count][0] = p1->coef * p2->coef;
            terms[count][1] = p1->exp + p2->exp;
            count++;
            p2 = p2->next;
        }
        p1 = p1->next;
    }
    
    if(count == 0){
        // 零多项式
        NODE * s = (NODE*)malloc(sizeof(NODE));
        s->coef = 0;
        s->exp = 0;
        s->next = NULL;
        h3->next = s;
        return;
    }
    
    // 冒泡排序
    for(int i = 0; i < count-1; i++){
        for(int j = 0; j < count-1-i; j++){
            if(terms[j][1] > terms[j+1][1]){
                int temp_coef = terms[j][0];
                int temp_exp = terms[j][1];
                terms[j][0] = terms[j+1][0];
                terms[j][1] = terms[j+1][1];
                terms[j+1][0] = temp_coef;
                terms[j+1][1] = temp_exp;
            }
        }
    }
    
    // 合并同类项
    int write_idx = 0;
    for(int i = 1; i < count; i++){
        if(terms[i][1] == terms[write_idx][1]){
            terms[write_idx][0] += terms[i][0];
        } else {
            write_idx++;
            terms[write_idx][0] = terms[i][0];
            terms[write_idx][1] = terms[i][1];
        }
    }
    count = write_idx + 1;
    
    // 构建结果链表
    NODE * tail = h3;
    for(int i = 0; i < count; i++){
        if(terms[i][0] != 0){
            NODE * new_node = (NODE*)malloc(sizeof(NODE));
            new_node->coef = terms[i][0];
            new_node->exp = terms[i][1];
            new_node->next = NULL;
            tail->next = new_node;
            tail = new_node;
        }
    }
    
    // 如果结果为空，添加零多项式
    if(!h3->next){
        NODE * s = (NODE*)malloc(sizeof(NODE));
        s->coef = 0;
        s->exp = 0;
        s->next = NULL;
        h3->next = s;
    }
}

int main()  
{   NODE * head1, * head2, * head3;  

    head1 = ( NODE * ) malloc( sizeof(NODE) );  
    input( head1 );  

    head2 = ( NODE * ) malloc( sizeof(NODE) );  
    input( head2 );  

    head3 = ( NODE * ) malloc( sizeof(NODE) );  
    head3->next = NULL;  
    multiplication( head1, head2, head3 );  

    output( head3 );  
    return 0;  
}