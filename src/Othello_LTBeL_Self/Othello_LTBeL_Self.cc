//********************************************************//
//  Created by Dhgir.Abien on 2018/7/4.                   //
//  Copyright © 2018年 Dhgir.Abien. All rights reserved.  //
//********************************************************//

#ifdef CNN
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#define Model_Path "./data/model.pb"
#endif

#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <functional>
#include <algorithm>
#include <future>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <fstream>
#include <random>
#include <ctime>
#define History_Path "./data/History"


using namespace std;

#define Board_Size 10
#define PASS -1
#define c_factor 10
#define Branching_Factor 100
#define Virtual_Loss 0.05
#define Virtual_policy_Loss 0.0005

enum player_markers:char{nihil, dark, light};
typedef struct Move_t{ char x, y; } Move_t;
const char dir_x[8] = {0, 1, 1, 1, 0, -1, -1, -1};
const char dir_y[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
atomic<int> MultiThread_NUM(100);
int Thread_Num = 1;
int Time_Limit = 120;
int Some_Expansion_Count = 0;
unsigned long long int Search_Loop_Count = 0;
int my_color;
int op_color;
bool value_Flag = false;
int value_Limit = 1;
vector<Move_t> history;
bool new_game_flag = 1;
bool pre_search_flag = false;
bool pre_search_stop = false;
bool pre_searching = false;
bool pre_search = false;
int toEnd_deep = 0;
int total_hand_number = 4;
double noise = 0.0000025;

random_device rd;
default_random_engine gen;
uniform_int_distribution<int> dis(0,100000);


class Timer {
public:
    Timer():tpStart(std::chrono::high_resolution_clock::now()),tpStop(tpStart)
    {
    }
    
    void start(){
        tpStart = std::chrono::high_resolution_clock::now();
        
    }
    void stop(){
        tpStop = std::chrono::high_resolution_clock::now();
        
    }
    template <typename span>
    int delta() const{
        return (int)(std::chrono::duration_cast<span>(std::chrono::high_resolution_clock::now() - tpStart).count());
    }
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> tpStart;
    std::chrono::time_point<std::chrono::high_resolution_clock> tpStop;
};

class State{
public:
    char now_board[ Board_Size ][ Board_Size ];
    char last_board[ Board_Size ][ Board_Size ];
    char legal_board[ Board_Size ][ Board_Size ];
    float policy_board[ Board_Size ][ Board_Size ];
    char legal_count;
    char dark_num;
    char light_num;
    char hand_number;
    char player_to_move;
    
    State():
    legal_count(0), dark_num(0), light_num(0), hand_number(4), player_to_move(dark)
    {
        memset(now_board, 0, sizeof(now_board[0][0]) * Board_Size * Board_Size);
        memset(last_board, 0, sizeof(last_board[0][0]) * Board_Size * Board_Size);
        memset(legal_board, 0, sizeof(legal_board[0][0]) * Board_Size * Board_Size);
        memset(policy_board, 0, sizeof(policy_board[0][0]) * Board_Size * Board_Size);
        
        now_board[4][4] = now_board[5][5] = light;
        now_board[4][5] = now_board[5][4] = dark;
        
        Find_Legal_Moves(player_to_move);
    }
    
    void do_Move(Move_t move){
        hand_number++;
        for(int i=0; i<Board_Size; i++)
            for(int j=0; j<Board_Size; j++)
                last_board[i][j] = now_board[i][j];
        
        if(move.x != PASS){
            now_board[move.x][move.y] = player_to_move;
            Check_Cross(move.x, move.y, true);
        }
        Be_ready();
    }
    
    void Be_ready(void){
        player_to_move = 3 - player_to_move;
        Find_Legal_Moves(player_to_move);
    }
    
    bool is_Legal(Move_t mv){
        return legal_board[mv.x][mv.y];
    }
    
    char get_Winner(){
        if(legal_count == 0){
            if(Find_Legal_Moves(3-player_to_move) == 0){
                return (dark_num > light_num) ? dark : light;
            }
            else{
                Find_Legal_Moves(player_to_move);
                return nihil;
            }
        }
        else
            return nihil;
    }
    
    int count_dark(){
        return dark_num;
    }
    
    int count_light(){
        return light_num;
    }
    
    int count_legal(){
        return legal_count;
    }
    
    Move_t get_best_move(){
        float bestP = 0;
        Move_t mv, bmv;
        for(mv.x=0; mv.x<Board_Size; mv.x++)
            for(mv.y=0; mv.y<Board_Size; mv.y++){
                if(legal_board[mv.x][mv.y] == true){
                    if(policy_board[mv.x][mv.y] > bestP){
                        bestP = policy_board[mv.x][mv.y];
                        bmv = mv;
                    }
                }
            }
        policy_board[bmv.x][bmv.y] += -1;
        return bmv;
    }
    
    void print_State(void){
        cout << "\n=========================" << endl;
        cout << "Hand Number: "<< (int)hand_number << endl;
        cout << "Turn "<< (int)player_to_move << " to move: " << endl;
        cout << "a b c d e f g h i j" << endl;
        for(int i=0; i<Board_Size; i++){
            for(int j=0; j<Board_Size; j++){
                if(now_board[j][i] == light)
                    cout << "O ";
                else if(now_board[j][i] == dark)
                    cout << "X ";
                else if (legal_board[j][i] == true)
                    cout << "? ";
                else
                    cout << ". ";
            }
            cout << " " << i+1 << endl;
        }
        cout << endl;
    }
    
private:
    int In_Board(int x, int y) const{
        if( x >= 0 && x < Board_Size && y >= 0 && y < Board_Size )
            return true;
        else
            return false;
    }
    
    int Check_Straight_Army(int x, int y, int d, int update){
        int me = now_board[x][y];
        int army = 3 - me;
        int army_count=0;
        int found_flag=false;
        int flag[ 10 ][ 10 ] = {{0}};
        
        int i, j;
        int tx, ty;
        
        tx = x;
        ty = y;
        
        for(i=0 ; i<Board_Size ; i++){
            tx += dir_x[d];
            ty += dir_y[d];
            
            if(In_Board(tx,ty) ){
                if( now_board[tx][ty] == army ){
                    army_count ++;
                    flag[tx][ty] = true;
                }
                else if( now_board[tx][ty] == me){
                    found_flag = true;
                    break;
                }
                else
                    break;
            }
            else
                break;
        }
        
        if( (found_flag == true) && (army_count > 0) && update){
            for(i=0 ; i<Board_Size ; i++)
                for(j=0; j<Board_Size ; j++)
                    if(flag[i][j]==true){
                        if(now_board[i][j] != 0)
                            now_board[i][j]= 3 - now_board[i][j];
                    }
        }
        if( (found_flag == true) && (army_count > 0))
            return army_count;
        else return 0;
    }
    
    int Check_Cross(int x, int y, int update){
        int k, dx, dy;
        
        if( ! In_Board(x,y) || now_board[x][y] == 0)
            return false;
        
        int army = 3 - now_board[x][y];
        int army_count = 0;
        
        for( k=0 ; k<8 ; k++ ){
            dx = x + dir_x[k];
            dy = y + dir_y[k];
            if( In_Board(dx,dy) && now_board[dx][dy] == army ){
                army_count += Check_Straight_Army( x, y, k, update ) ;
            }
        }
        
        if(army_count >0)
            return true;
        else
            return false;
    }
    
    int Find_Legal_Moves( int color ){
        int i,j;
        int me = color;
        dark_num = light_num = 0;
        
        for( i = 0; i < Board_Size; i++ )
            for( j = 0; j < Board_Size; j++ ){
                legal_board[i][j] = 0;
            }
        legal_count = 0;
        
        for( i = 0; i < Board_Size; i++ )
            for( j = 0; j < Board_Size; j++ ){
                if( now_board[i][j] == 0 ){
                    if( i>0 && i<Board_Size-1 && j>0 && j<Board_Size-1 ){
                        if((now_board[i-1][j-1]  == 0 || now_board[i-1][j-1] == me) &&
                           (now_board[i-1][j]    == 0 || now_board[i-1][j]   == me) &&
                           (now_board[i-1][j+1]  == 0 || now_board[i-1][j+1] == me) &&
                           (now_board[i][j-1]    == 0 || now_board[i][j-1]   == me) &&
                           (now_board[i][j+1]    == 0 || now_board[i][j+1]   == me) &&
                           (now_board[i+1][j-1]  == 0 || now_board[i+1][j-1] == me) &&
                           (now_board[i+1][j]    == 0 || now_board[i+1][j]   == me) &&
                           (now_board[i+1][j+1]  == 0 || now_board[i+1][j+1] == me)){
                            continue;
                        }
                    }
                    now_board[i][j] = me;
                    if( Check_Cross(i,j,false) == true ){
                        legal_board[i][j] = true;
                        legal_count++;
                    }
                    now_board[i][j] = 0;
                }
                else if(now_board[i][j] == dark)
                    dark_num++;
                else if(now_board[i][j] == light)
                    light_num++;
            }
        return legal_count;
    }
};


class Node{
public:
    float value;
    float policy;
    int visit;
    float uct;
    bool is_solved;
    Move_t mv;
    Node* ancestor;
    vector<Node*> min_queue;
    
    Node():
    value(0), policy(0), visit(1), uct(0), is_solved(false), ancestor(nullptr)
    {
    }
    
    Node(Move_t in_mv, Node* in_ancestor):
    value(0), policy(0), visit(1), uct(0), is_solved(false), mv(in_mv), ancestor(in_ancestor)
    {
    }
    
private:
    
};


class Estimate{
public:
    Estimate():
    return_flag(true),return_count(0)
    {
#ifdef CNN
        tensorflow::Status status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
        string graph_path = Model_Path;
        status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graph_path, &graph_def);
        status = session->Create(graph_def);
#endif      
    }
    
#ifdef CNN
    void value_Simulation_Thread(void){
        for(;;){
            unique_lock<mutex> lk(val_queue_mut);
            while(!(val_simulation_queue.size() == MultiThread_NUM && val_ready_flag == false)){
                val_prepare.wait_for(lk ,std::chrono::milliseconds(1));
            }
            
            int len = 0;
            
            for(int i=0; i<val_simulation_queue.size();i++){
                if(val_simulation_state[i] == nullptr){
                    continue;
                }
                total_val_simulation_queue.push_back(val_simulation_queue[i]);
                total_val_simulation_state.push_back(*val_simulation_state[i]);
            }
            
            if(total_hand_number >= 100-toEnd_deep || value_Flag != true){
                val_ready_flag = true;
                return_count = (int)val_simulation_queue.size();
                return_flag = false;
                val_simulation_queue.clear();
                val_simulation_state.clear();
                val_ready.notify_all();
                continue;
            }
            else{
                for(int k=0; k<total_val_simulation_queue.size(); k++){
                    if(total_val_simulation_queue[k] != nullptr)
                        len += total_val_simulation_queue[k]->size();
                }
            }
            
            tensorflow::Tensor inin(tensorflow::DT_FLOAT, tensorflow::TensorShape({len, 10, 10, 5}));
            auto inin_mapped = inin.tensor<float, 4>();
            State temp_state;
            
            int index = 0;
            for(int k=0; k<total_val_simulation_queue.size(); k++){
                if(total_val_simulation_queue[k] != nullptr){
                    
                    for(int v=0; v<total_val_simulation_queue[k]->size(); v++, index++){
                        temp_state = total_val_simulation_state[k];
                        temp_state.do_Move(total_val_simulation_queue[k]->at(v)->mv);
                        
                        for(int i=0; i<Board_Size; i++)
                            for(int j=0; j<Board_Size; j++){
                                if( temp_state.now_board[j][i] == 1 ){
                                    inin_mapped(index, i, j, 0) = 1.0f;
                                    inin_mapped(index, i, j, 2) = 0.0f;
                                }
                                else if( temp_state.now_board[j][i] == 2 ){
                                    inin_mapped(index, i, j, 0) = 0.0f;
                                    inin_mapped(index, i, j, 2) = 1.0f;
                                }
                                else{
                                    inin_mapped(index, i, j, 0) = 0.0f;
                                    inin_mapped(index, i, j, 2) = 0.0f;
                                }
                            }
                        
                        for(int i=0; i<Board_Size; i++)
                            for(int j=0; j<Board_Size; j++){
                                if( temp_state.last_board[j][i] == 1 ){
                                    inin_mapped(index, i, j, 1) = 1.0f;
                                    inin_mapped(index, i, j, 3) = 0.0f;
                                }
                                else if( temp_state.last_board[j][i] == 2 ){
                                    inin_mapped(index, i, j, 1) = 0.0f;
                                    inin_mapped(index, i, j, 3) = 1.0f;
                                }
                                else{
                                    inin_mapped(index, i, j, 1) = 0.0f;
                                    inin_mapped(index, i, j, 3) = 0.0f;
                                }
                            }
                        
                        for(int i=0; i<Board_Size; i++)
                            for(int j=0; j<Board_Size; j++){
                                if( temp_state.legal_board[j][i] == 1 ){
                                    inin_mapped(index, i, j, 4) = 1.0f;
                                }
                                else
                                    inin_mapped(index, i, j, 4) = 0.0f;
                            }
                    }
                }
                
            }
            
            
            std::vector<std::pair<string, tensorflow::Tensor>> inputs = {{ "input_4", inin }};
            std::vector<tensorflow::Tensor> outputs;
            tensorflow::Status status = session->Run(inputs, {"output_1"}, {}, &outputs);
            tensorflow::Tensor* output = &outputs[0];
            const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>, Eigen::Aligned>& prediction = output->flat<float>();
            
            index = 0;
            for(int k=0; k<total_val_simulation_queue.size(); k++){
                if(total_val_simulation_queue[k] != nullptr){
                    for(int v=0; v<total_val_simulation_queue[k]->size(); v++, index++){
                        if(my_color == 1)
                            total_val_simulation_queue[k]->at(v)->value = 0.75*prediction(index)+noise*dis(gen) ;
                        else
                            total_val_simulation_queue[k]->at(v)->value = -1*(0.75*prediction(index)+noise*dis(gen));
                    }
                }
            }
            
            val_ready_flag = true;
            return_count = (int)val_simulation_queue.size();
            return_flag = false;
            val_simulation_queue.clear();
            val_simulation_state.clear();
            total_val_simulation_queue.clear();
            total_val_simulation_state.clear();
            val_ready.notify_all();
            
        }
    }
    
    void policy_Simulation_Thread(void){
        for(;;){
            unique_lock<mutex> lk(pol_queue_mut);
            while(!(pol_simulation_queue.size() == MultiThread_NUM && pol_ready_flag == false)){
                pol_prepare.wait_for(lk ,std::chrono::milliseconds(1));
            }
            
            int len = MultiThread_NUM;
            
            tensorflow::Tensor inin(tensorflow::DT_FLOAT, tensorflow::TensorShape({len, 10, 10, 5}));
            auto inin_mapped = inin.tensor<float, 4>();
            
            for(int k=0; k<len; k++){
                if(pol_simulation_queue[k] != nullptr){
                    for(int i=0; i<Board_Size; i++)
                        for(int j=0; j<Board_Size; j++){
                            if( pol_simulation_queue[k]->now_board[j][i] == 1 ){
                                inin_mapped(k, i, j, 0) = 1.0f;
                                inin_mapped(k, i, j, 2) = 0.0f;
                            }
                            else if( pol_simulation_queue[k]->now_board[j][i] == 2 ){
                                inin_mapped(k, i, j, 0) = 0.0f;
                                inin_mapped(k, i, j, 2) = 1.0f;
                            }
                            else{
                                inin_mapped(k, i, j, 0) = 0.0f;
                                inin_mapped(k, i, j, 2) = 0.0f;
                            }
                        }
                    
                    for(int i=0; i<Board_Size; i++)
                        for(int j=0; j<Board_Size; j++){
                            if( pol_simulation_queue[k]->last_board[j][i] == 1 ){
                                inin_mapped(k, i, j, 1) = 1.0f;
                                inin_mapped(k, i, j, 3) = 0.0f;
                            }
                            else if( pol_simulation_queue[k]->last_board[j][i] == 2 ){
                                inin_mapped(k, i, j, 1) = 0.0f;
                                inin_mapped(k, i, j, 3) = 1.0f;
                            }
                            else{
                                inin_mapped(k, i, j, 1) = 0.0f;
                                inin_mapped(k, i, j, 3) = 0.0f;
                            }
                        }
                    
                    for(int i=0; i<Board_Size; i++)
                        for(int j=0; j<Board_Size; j++){
                            if( pol_simulation_queue[k]->legal_board[j][i] == 1 ){
                                inin_mapped(k, i, j, 4) = 1.0f;
                            }
                            else
                                inin_mapped(k, i, j, 4) = 0.0f;
                        }
                }
            }
            
            std::vector<std::pair<string, tensorflow::Tensor>> inputs = {{ "input_4", inin }};
            std::vector<tensorflow::Tensor> outputs1;
            tensorflow::Status status = session->Run(inputs, {"output_0"}, {}, &outputs1);
            tensorflow::Tensor* output1 = &outputs1[0];
            const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>, Eigen::Aligned>& prediction1 = output1->flat<float>();
            
            for(int k=0; k<len; k++){
                if(pol_simulation_queue[k] != nullptr){
                    for(int i=0; i<Board_Size; i++){
                        for(int j=0; j<Board_Size; j++){
                            pol_simulation_queue[k]->policy_board[j][i] = prediction1(k*100+i*10+j);
                        }
                    }
                }
            }
            
            pol_ready_flag = true;
            return_count = (int)pol_simulation_queue.size();
            return_flag = false;
            pol_simulation_queue.clear();
            pol_ready.notify_all();
            
        }
    }
#endif
    
#ifndef CNN
    void value_Simulation_Thread(void){
        for(;;){
            unique_lock<mutex> lk(val_queue_mut);
            while(!(val_simulation_queue.size() == MultiThread_NUM && val_ready_flag == false)){
                val_prepare.wait_for(lk ,std::chrono::milliseconds(1));
            }
            
            int len = 0;
            for(int i=0; i<val_simulation_queue.size();i++){
                if(val_simulation_state[i] == nullptr){
                    continue;
                }
                total_val_simulation_queue.push_back(val_simulation_queue[i]);
                total_val_simulation_state.push_back(*val_simulation_state[i]);
            }
            
            if(total_hand_number >= 100-toEnd_deep || value_Flag != true){
                val_ready_flag = true;
                return_count = (int)val_simulation_queue.size();
                return_flag = false;
                val_simulation_queue.clear();
                val_simulation_state.clear();
                val_ready.notify_all();
                continue;
            }
            else{
                for(int k=0; k<total_val_simulation_queue.size(); k++){
                    if(total_val_simulation_queue[k] != nullptr)
                        len += total_val_simulation_queue[k]->size();
                }
            }
            
            for(int k=0; k<total_val_simulation_queue.size(); k++){
                if(total_val_simulation_queue[k] != nullptr){
                    for(int v=0; v<total_val_simulation_queue[k]->size(); v++){
                        Node** ptr = total_val_simulation_queue[k]->data();
                        if(ptr != nullptr && ptr[0]->ancestor != nullptr)
                            total_val_simulation_queue[k]->at(v)->value = (float)(rand()) / RAND_MAX;
                    }
                }
            }
            
            val_ready_flag = true;
            return_count = (int)val_simulation_queue.size();
            return_flag = false;
            val_simulation_queue.clear();
            val_simulation_state.clear();
            total_val_simulation_queue.clear();
            total_val_simulation_state.clear();
            val_ready.notify_all();
        }
    }
    
    void policy_Simulation_Thread(void){
        for(;;){
            unique_lock<mutex> lk(pol_queue_mut);
            while(!(pol_simulation_queue.size() == MultiThread_NUM && pol_ready_flag == false)){
                pol_prepare.wait_for(lk ,std::chrono::milliseconds(1));
            }
            
            for(int k=0; k<MultiThread_NUM; k++){
                for(int i=0; i<Board_Size; i++){
                    for(int j=0; j<Board_Size; j++){
                        if(pol_simulation_queue[k] != nullptr){
                            pol_simulation_queue[k]->policy_board[i][j] = (float)(rand()) / RAND_MAX;
                        }
                    }
                }
            }
            
            pol_ready_flag = true;
            return_count = (int)pol_simulation_queue.size();
            return_flag = false;
            pol_simulation_queue.clear();
            pol_ready.notify_all();
        }
    }
#endif
    
    void set_Policy(State &simulation_state, int servant_id){
        while(return_flag == false){
            if(return_count == 0)
                return_flag = true;
            continue;
        }
        
        pol_queue_mut.lock();
        pol_simulation_queue.push_back(&simulation_state);
        pol_ready_flag = false;
        pol_queue_mut.unlock();
        
        pol_prepare.notify_one();
        
        unique_lock<mutex> lk(pol_queue_mut);
        pol_ready.wait(lk,[this]{return pol_ready_flag == true;});
        
        if(--return_count == 0)
            return_flag = true;
    }
    
    void set_Pol_Pass(int servant_id){
        while(return_flag == false)
            continue;
        
        pol_queue_mut.lock();
        pol_simulation_queue.push_back(nullptr);
        pol_ready_flag = false;
        pol_queue_mut.unlock();
        
        pol_prepare.notify_one();
        
        unique_lock<mutex> lk(pol_queue_mut);
        pol_ready.wait(lk,[this]{return pol_ready_flag == true;});
        
        if(--return_count == 0)
            return_flag = true;
    }
    
    void set_Value(vector<Node*> *simulation_queue ,State *root_state, int servant_id){
        while(return_flag == false)
            continue;
        
        val_queue_mut.lock();
        val_simulation_queue.push_back(simulation_queue);
        val_simulation_state.push_back(root_state);
        val_ready_flag = false;
        val_queue_mut.unlock();
        
        val_prepare.notify_one();
        
        unique_lock<mutex> lk(val_queue_mut);
        val_ready.wait(lk,[this]{return val_ready_flag == true;});
        
        if(--return_count == 0)
            return_flag = true;
    }
    
    void set_Val_Pass(int servant_id){
        while(return_flag == false)
            continue;
        
        val_queue_mut.lock();
        val_simulation_queue.push_back(nullptr);
        val_simulation_state.push_back(nullptr);
        val_ready_flag = false;
        val_queue_mut.unlock();
        
        val_prepare.notify_one();
        
        unique_lock<mutex> lk(val_queue_mut);
        val_ready.wait(lk,[this]{return val_ready_flag == true;});
        
        if(--return_count == 0)
            return_flag = true;
    }
    
    void re_Check(void){
        val_prepare.notify_one();
        pol_prepare.notify_one();
    }
    
private:
#ifdef CNN
    tensorflow::Session* session;
    tensorflow::GraphDef graph_def;
#endif
    
    atomic<int> return_count;
    atomic<bool> return_flag;
    
    mutex pol_queue_mut;
    vector<State*> pol_simulation_queue;
    condition_variable pol_prepare;
    condition_variable pol_ready;
    atomic<bool> pol_ready_flag;
    
    mutex val_queue_mut;
    vector<vector<Node*>*> val_simulation_queue;
    vector<State*> val_simulation_state;
    condition_variable val_prepare;
    condition_variable val_ready;
    atomic<bool> val_ready_flag;
    
    vector<vector<Node*>*> total_val_simulation_queue;
    vector<State> total_val_simulation_state;
    
};
Estimate *myCNN;


class Servant{
public:
    Servant(int id):
    Servant_ID(id)
    {
    }
    
    Node* expansion(Node** root, State now_state){
        Node** select_node = root;
        
        while((*select_node)->min_queue.empty() == false){
            select_node = &(*min_element((*select_node)->min_queue.begin(),(*select_node)->min_queue.end(),
                                         [](Node* a, Node* b) {
                                             if(a->is_solved == b->is_solved) return a->uct < b->uct;
                                             else return !a->is_solved;
                                         }));
            now_state.do_Move((*select_node)->mv);
            ++(*select_node)->visit;
            (*select_node)->uct += ((*select_node)->value == 0)?Virtual_policy_Loss:Virtual_Loss;
        }
        
        Node* now_node = new Node;
        *now_node = **select_node;
        
        if(now_state.legal_count == 0){
            myCNN->set_Pol_Pass(Servant_ID);
            
            if(now_state.get_Winner() == nihil){
                Node* temp = new Node;
                temp->mv.x = temp->mv.y = PASS;
                temp->ancestor = now_node;
                temp->policy = now_node->policy;
                now_node->min_queue.push_back(temp);
            }
            else{
                Node* temp = new Node;
                temp->mv.x = temp->mv.y = PASS;
                temp->ancestor = now_node;
                temp->is_solved = true;
                
                if(now_state.count_dark() == now_state.count_light())
                    temp->value = 0;
                else if(now_state.count_dark() > now_state.count_light()){
                    temp->value = now_state.count_dark() - now_state.count_light();
                    temp->value *= (now_state.player_to_move == dark) ? -1:1;
                }
                else{
                    temp->value = now_state.count_light() - now_state.count_dark();
                    temp->value *= (now_state.player_to_move == dark) ? 1:-1;
                }
                
                now_node->min_queue.push_back(temp);
            }
        }
        else{
            myCNN->set_Policy(now_state, Servant_ID);
            
            for(int i=0; i<Branching_Factor && i<now_state.legal_count; i++){
                Node* temp = new Node;
                temp->mv = now_state.get_best_move();
                temp->ancestor = now_node;
                temp->policy = now_state.policy_board[temp->mv.x][temp->mv.y]+1;
                now_node->min_queue.push_back(temp);
            }
        }
        
        for(int i=0; i<now_node->min_queue.size(); i++){
            State temp_state = now_state;
            Node *temp_node = now_node->min_queue[i];
            temp_state.do_Move(temp_node->mv);
            
            if(temp_state.legal_count == 0){
                myCNN->set_Pol_Pass(Servant_ID);
                myCNN->set_Val_Pass(Servant_ID);
                
                if(temp_state.get_Winner() == nihil){
                    Node* temp = new Node;
                    temp->mv.x = temp->mv.y = PASS;
                    temp->ancestor = temp_node;
                    temp->policy = temp_node->policy;
                    temp_node->min_queue.push_back(temp);
                }
                else{
                    Node* temp = new Node;
                    temp->mv.x = temp->mv.y = PASS;
                    temp->ancestor = temp_node;
                    temp->is_solved = true;
                    
                    if(temp_state.count_dark() == temp_state.count_light())
                        temp->value = 0;
                    else if(temp_state.count_dark() > temp_state.count_light()){
                        temp->value = temp_state.count_dark() - temp_state.count_light();
                        temp->value *= (temp_state.player_to_move == dark) ? -1:1;
                    }
                    else{
                        temp->value = temp_state.count_light() - temp_state.count_dark();
                        temp->value *= (temp_state.player_to_move == dark) ? 1:-1;
                    }
                    
                    temp_node->min_queue.push_back(temp);
                }
            }
            else{
                myCNN->set_Policy(temp_state, Servant_ID);
                for(int j=0; j<Branching_Factor && j<temp_state.legal_count; j++){
                    Node* temp = new Node;
                    temp->mv = temp_state.get_best_move();
                    temp->ancestor = temp_node;
                    temp->policy = temp_state.policy_board[temp->mv.x][temp->mv.y]+1;
                    temp_node->min_queue.push_back(temp);
                }
                
                myCNN->set_Value(&temp_node->min_queue ,&temp_state, Servant_ID);
                
                for(int j=0; j<temp_node->min_queue.size(); j++){
                    temp_node->min_queue[j]->uct = temp_node->min_queue[j]->value - c_factor * temp_node->min_queue[j]->policy * sqrt(float(temp_node->min_queue[j]->ancestor->visit))/(temp_node->min_queue[j]->visit);
                }
            }
        }
        
        --MultiThread_NUM;
        myCNN->re_Check();
        
        if((*select_node)->min_queue.empty() == true){
            
            *select_node = now_node;
        }
        else{
            delete_node(now_node);
            Some_Expansion_Count++;
        }
        return *select_node;
    }
    
private:
    int Servant_ID;
    
    void delete_node(Node * d_node){
        if(d_node == nullptr)
            return;
        while(d_node->min_queue.empty() == false){
            delete_node( d_node->min_queue.back());
            d_node->min_queue.pop_back();
        }
        delete d_node;
        d_node = nullptr;
    }
};

class Master{
public:
    
    Master()
    {
        root = new Node;
        root->mv.x = 25;
        root->mv.y = -1;
        Search_Loop_Count = 0;
        Total_think_time = 0;
        Total_Search_Loop_Count = 0;
        Backtrack_Count = 0;
        debug_Count = 0;
        value_Count = 0;

        gen = default_random_engine(rd());
        
        for(int i=0; i<Thread_Num; i++)
            casper[i] = new Servant(i);
    }
    
    unsigned long long int Total_think_time;
    unsigned long long int Total_Search_Loop_Count;
    unsigned long long int Backtrack_Count;
    
    Move_t get_Best_Move(State root_state){
        loopTimer.start();
        is_Sloved_flag = false;
        value_Flag = false;
        value_Count = 0;
        pre_searching = true;
        
        while(is_Time_Out() == false){
            if(++value_Count == value_Limit){
                value_Count = 0;
                value_Flag = true;
            }
            
            if(root->min_queue.empty() == true || Thread_Num == 1){
                MultiThread_NUM = 1;
                Node* temp = casper[0]->expansion(&root, root_state);
                for(int i=0; i<temp->min_queue.size(); i++){
                    backtrack.push_back(temp->min_queue[i]);
                }
                Search_Loop_Count += temp->min_queue.size();
            }
            else{
                MultiThread_NUM = Thread_Num;
                for(int t=0; t<Thread_Num; t++){
                    expansion_threads.push_back(async(launch::async, &Servant::expansion, casper[t], &root, root_state));
                }
                
                for(int t=0; t<Thread_Num; t++){
                    Node* temp = move(expansion_threads[t].get());
                    for(int i=0; i<temp->min_queue.size(); i++){
                        backtrack.push_back(temp->min_queue[i]);
                    }
                    Search_Loop_Count += temp->min_queue.size();
                }
                
            }
            
            if(value_Flag == false){
                expansion_threads.clear();
                continue;
            }
            
            Backtrack_Count += backtrack.size();
            for(int t=0; t<backtrack.size(); t++){
                Node* now_node = backtrack[t];
                
                while(now_node != nullptr){
                    auto min_node = *min_element(now_node->min_queue.begin(), now_node->min_queue.end(),
                                                 [](Node* a, Node* b) { return a->value < b->value; });
                    now_node->value = -min_node->value;
                    
                    if(now_node != root){
                        now_node->uct = now_node->value - c_factor * now_node->policy * sqrt(float(now_node->ancestor->visit))/(now_node->visit);
                    }
                    
                    now_node->is_solved = true;
                    for(int i=0; i<now_node->min_queue.size(); i++)
                        if(now_node->min_queue[i]->is_solved == false){
                            now_node->is_solved = false;
                            break;
                        }
                    
                    now_node = now_node->ancestor;
                }
            }
            backtrack.clear();
            expansion_threads.clear();
        }
        
        Node* ans_node = nullptr;
        int max_num = 100;
        if(is_Sloved_flag == true){
            for(int i=0; i<root->min_queue.size(); i++){
                if(root->min_queue[i]->is_solved == 1 && root->min_queue[i]->value < max_num){
                    ans_node = root->min_queue[i];
                    max_num = root->min_queue[i]->value;
                }
            }
        }
        else{
            ans_node = *min_element(root->min_queue.begin(), root->min_queue.end(),
                                    [](Node* a, Node* b) { return a->value < b->value; });
        }
        
        if(pre_search_flag == false)
            optimize_memory(ans_node->mv);
        
        pre_searching = false;
        return ans_node->mv;
    }
    
    void optimize_memory(Move_t select_mv){
        unsigned long int len = root->min_queue.size();
        Node* select_node = nullptr;
        for(int i=0; i<len; i++){
            if(root->min_queue[i]->mv.x != select_mv.x || root->min_queue[i]->mv.y != select_mv.y){
                delete_node(root->min_queue[i]);
            }
            else{
                select_node = root->min_queue[i];
            }
        }
        
        if(select_node == nullptr){
            select_node = new Node;
            select_node->mv = select_mv;
        }
        
        delete root;
        select_node->ancestor = nullptr;
        root = select_node;
    }
    
    
private:
    Node* root;
    Servant *casper[128];
    vector<Node*> backtrack;
    vector<future<Node*>> expansion_threads;
    Timer loopTimer;
    bool is_Sloved_flag;
    unsigned int debug_Count;
    unsigned int value_Count;
    
    bool is_Time_Out(){
        debug_Count++;
        
#ifdef Debug
        if(debug_Count%1 == 0){
            if(pre_search_flag == false){
                cout << "\n#############" << endl;
                for(int i=0; i<root->min_queue.size(); i++){
                    if(root->min_queue[i]->mv.x != PASS)
                        cout << "Now Move: " << (char)(root->min_queue[i]->mv.x+'a') << root->min_queue[i]->mv.y+1 << endl;
                    else
                        cout << "Now Move: ps";
                    cout << "  Value: " << root->min_queue[i]->value << endl;
                    cout << "  Policy: " << root->min_queue[i]->policy << endl;
                    cout << "  UCT: " << root->min_queue[i]->uct << endl;
                    cout << "  visits: " << root->min_queue[i]->visit << endl;
                    cout << "  is_sloved: " << root->min_queue[i]->is_solved << endl;
                }
                cout << "\n#Some_Expansion: " << Some_Expansion_Count << endl;
                cout << "#Search_Loop_Count: " << Search_Loop_Count << endl;
                cout << "#toEnd_flag: " << (bool)(total_hand_number >= 100-toEnd_deep) << endl;
                
                Node* tempNode = root;
                cout << "#Best_Line:";
                while(tempNode->min_queue.empty()  == false){
                    if(tempNode->mv.x != PASS)
                        cout << " " << (char)(tempNode->mv.x+'a') << tempNode->mv.y+1;
                    else
                        cout << " ps";
                    tempNode = *min_element(tempNode->min_queue.begin(), tempNode->min_queue.end(),
                                            [](Node* a, Node* b) { return a->value < b->value; });
                }
                cout << endl << endl;
            }
            else
                cout << '.' << endl;
        }
#endif
        for(int i=0; i<root->min_queue.size(); i++){
            if(root->min_queue[i]->is_solved == 1 && root->min_queue[i]->value <= 0){
                is_Sloved_flag = true;
                break;
            }
        }
        
        if(pre_search_flag == true && pre_search_stop == false){
            value_Flag = false;
            return false;
        }
        else if(pre_search_stop == false && root->is_solved == false && is_Sloved_flag == false &&
                loopTimer.delta<std::chrono::seconds>() < Time_Limit ){
            value_Flag = false;
            return false;
        }
        else if(value_Flag == false){
            value_Flag = true;
            return false;
        }
        else{
            if(pre_search_flag == false){
                cout << "#############" << endl;
                for(int i=0; i<root->min_queue.size(); i++){
                    if(root->min_queue[i]->mv.x != PASS)
                        cout << "Now Move: " << (char)(root->min_queue[i]->mv.x+'a') << root->min_queue[i]->mv.y+1 << endl;
                    else
                        cout << "Now Move: ps";
                    cout << "  Value: " << root->min_queue[i]->value << endl;
                    cout << "  Policy: " << root->min_queue[i]->policy << endl;
                    cout << "  UCT: " << root->min_queue[i]->uct << endl;
                    cout << "  visits: " << root->min_queue[i]->visit << endl;
                    cout << "  is_sloved: " << root->min_queue[i]->is_solved << endl;
                }
                cout << "\n#Some_Expansion: "<< Some_Expansion_Count << endl;
                cout << "#Search_Loop_Count: " << Search_Loop_Count << endl;
                cout << "#value_Flag: " << value_Flag << endl;
                cout << "#toEnd_flag: " << (bool)(total_hand_number >= 100-toEnd_deep) << endl;
                
                Node* tempNode = root;
                cout << "#Best_Line:";
                while(tempNode->min_queue.empty()  == false){
                    if(tempNode->mv.x != PASS)
                        cout << " " << (char)(tempNode->mv.x+'a') << tempNode->mv.y+1;
                    else
                        cout << " ps";
                    
                    tempNode = *min_element(tempNode->min_queue.begin(), tempNode->min_queue.end(),
                                            [](Node* a, Node* b) { return a->value < b->value; });
                }
                cout << endl << endl;
            }
            
            value_Flag = false;
            Total_Search_Loop_Count += Search_Loop_Count;
            Search_Loop_Count = 0;
            Total_think_time += loopTimer.delta<std::chrono::seconds>();
            Some_Expansion_Count = 0;
            debug_Count = 0;
            return true;
        }
    }
    
    void delete_node(Node * d_node){
        while(d_node->min_queue.empty() == false){
            delete_node( d_node->min_queue.back());
            d_node->min_queue.pop_back();
        }
        delete d_node;
    }
    
};


int main(int argc, const char * argv[]) {
    
    cout << "##############################" << endl;
    cout << "##                          ##" << endl;
    cout << "##          Ollehto         ##" << endl;
    cout << "##      by Dhgir.Abien      ##" << endl;
    cout << "##                          ##" << endl;
    cout << "##############################" << endl << endl << endl;
    
    int op_color;
    op_color = 0;
    my_color = 3-op_color;
    Thread_Num = thread::hardware_concurrency();
    
    Time_Limit /= 50;
    
    value_Limit = 1;
    
    toEnd_deep = 90;
    pre_search = 0;
    new_game_flag = 1;
    
    if(new_game_flag == 0){
        for(;;){
            Move_t temp;
            char in_ch;
            int in_num;
            
            cin >> in_ch >> in_num;
            if(in_ch == 'z' && in_num == 0)
                break;
            
            temp.x = in_ch-'a';
            temp.y = in_num-1;
            history.push_back(temp);
        }
    }
    
    myCNN = new Estimate;
    thread VST(&Estimate::value_Simulation_Thread ,myCNN);
    thread PST(&Estimate::policy_Simulation_Thread ,myCNN);
    
    Master master;    
    State now_state;
    Move_t move;
    
    srand(time(NULL));
    
    if(new_game_flag == 0){
        for(int i=0; i<history.size(); i++){
            now_state.do_Move(history[i]);
        }
    }
    
    cout << "Game Start!!!" << endl;
    
    while(now_state.get_Winner() == nihil){
        now_state.print_State();
        total_hand_number = now_state.hand_number;
        
        if(op_color == nihil || now_state.player_to_move == my_color){
            
            move = master.get_Best_Move(now_state);
            
            now_state.do_Move(move);
            history.push_back(move);
            
            if(move.x != PASS)
                cout << "Computer palyed " << (char)(move.x+'a') << move.y+1 << endl;
            else
                cout << "Now Move: ps" << endl;
            cout << "=========================" << endl;
            
            if(pre_search == true){
                pre_search_flag = true;
                thread preS(&Master::get_Best_Move, &master, now_state);
                preS.detach();
            }
        }
        else{
            do{
                cout << "Please input move:" << endl;
                char in_ch;
                int in_num;
                cin >> in_ch >> in_num;
                move.x = in_ch-'a';
                move.y = in_num-1;
                
                if(now_state.count_legal() == 0){
                    move.x = PASS;
                    break;
                }
            }while(now_state.is_Legal(move) == false);
            
            if(pre_search == true){
                pre_search_stop = true;
                while(pre_searching == true){
                    cout << endl;
                    continue;
                }
                pre_search_flag = false;
                pre_search_stop = false;
            }
            
            now_state.do_Move(move);
            history.push_back(move);
            
            master.optimize_memory(move);
        }
        
        cout << "history:" << endl;
        for(int i=0; i<history.size(); i++){
            if(history[i].x != PASS)
                cout << " " << (char)(history[i].x+'a') << history[i].y+1;
            else
                cout << " ps";
        }
        cout << endl;
    }
    
    now_state.print_State();
    cout << "Dark :" << now_state.count_dark() << endl;
    cout << "light :" << now_state.count_light() << endl;
    if(now_state.count_dark() > now_state.count_light()){
        cout << "dark Win" << endl;
    }
    else if(now_state.count_dark() < now_state.count_light()){
        cout << "light Win" << endl;
    }
    else{
        cout << "draw" << endl;
    }
    
    cout << "Game Over" << endl << endl;
    cout << "Backtrack_Count:" << master.Backtrack_Count << endl;
    cout << "Total_Search_Loop_Count:" << master.Total_Search_Loop_Count << endl;
    cout << "used thinking time: "<< master.Total_think_time << " sec" << endl;
    

    fstream file;
    file.open( History_Path , ios::app);
    for(int i=0; i<history.size(); i++){
        if(history[i].x != PASS)
            file << " " << (char)(history[i].x+'a') << history[i].y+1;
        else
            file << " ps";
    }
    file << endl;
    file.close();
    
    return 0;
}

