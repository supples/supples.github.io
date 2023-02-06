---
layout: post
title: AI(인공지능) 기반 챗봇 서비스
subtitle: 한이음 멘토링 결과물
categories: AI&SW
tags: [chatbot]
---


{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "toc_visible": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "#**AI(인공지능) 기반 챗봇 서비스**"
      ],
      "metadata": {
        "id": "BnfroGEgWncG"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **프로젝트 소개**"
      ],
      "metadata": {
        "id": "o6TSJj1xW2FK"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "주제 : 챗봇을 이용한 서울 관광지 추천봇\n",
        "\n",
        "서울의 행정구별 관광지에 대한 정보를 제공하여 사용자들에게 신선한 체험의 기회를 제공하기 위해 챗봇을 제작하였다. 이를 통해 많은 사람들이 알지 못했던 장소를 알릴 수 있는 효과를 기대할 수 있다. 사용자 별로 특색에 맞는 관광지를 추천해 주며, 관광지를 찾아야 하는 불필요한 노동과 수고, 시간낭비를 덜 수 있다.\n",
        "\n"
      ],
      "metadata": {
        "id": "kAAqgCRnWYt8"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **DATA**"
      ],
      "metadata": {
        "id": "-OUuHjk2W6WD"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **오픈API 및 웹크롤링을 이용해 데이터 제작**\n",
        "\n",
        "공공데이터포털에서 제공하는 '한국관광공사 국문 관광정보 서비스_GW' 오픈 API를 이용해 사용할 데이터 제작하며 추가적으로 필요한 데이터는 웹크롤링을 이용한다.\n",
        "\n",
        "![image-20230206144321659](https://ifh.cc/g/5CCn0F.png)\n",
        "\n",
        "\n",
        "\n",
        "![img](https://ifh.cc/g/Roy5no.jpg) <br/>\n",
        "여러 데이터를 종합해 만든 데이터 모음\n",
        "\n"
      ],
      "metadata": {
        "id": "BeekP-dwW9ht"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **챗봇 소개**\n",
        "\n",
        "- 만들어진 데이터를 활용하여 카테고리별 챗봇을 제작.\n",
        "\n",
        "- 여러 카테고리의 챗봇을 하나의 앱에서 사용할 수 있도록 앱 제작.\n",
        "\n",
        "- 카테고리는 대분류 중분류 소분류 로 구분한다.\n",
        "\n",
        "- 대분류에는 문화예술/자연명소/레포츠/쇼핑/음식점/숙박시설이 있다. \n",
        "\n",
        "- 문화예술은 다시 건축.조형물/산업관광지/역사관광지/체험관광지/휴양관광지/문화시설로 나뉜다.\n",
        "\n",
        "- 자연명소는 강.계곡.폭포/공원 으로 나뉜다.\n",
        "\n",
        "  "
      ],
      "metadata": {
        "id": "uLpK15wPX_SB"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **작품 구상도**"
      ],
      "metadata": {
        "id": "2595GmqvYFRy"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### SW 구성도\n",
        "\n",
        "- Dialog Flow를 이용 : 구글의 머신러닝기술을 활용하여 사용자의 니즈를 예측하여 그에 맞는 질문과 대답들을 업데이트한다.\n",
        "\n",
        "![img](https://ifh.cc/g/q8WZg8.png)\n",
        "\n",
        "- ngrok 이용 페이스북 메시지 등과 같이 외부 서버와 연결하여 외부 사용자가 이용가능하게 한다.\n",
        "\n",
        "![img](https://th.bing.com/th/id/OIP.4pRFlMrovybUI_RxbXjj1AHaCs?w=348&h=127&c=7&r=0&o=5&dpr=1.5&pid=1.7)\n",
        "\n",
        "![img](https://ifh.cc/g/czNTqk.png) <br/>\n",
        "Dialog Flow와 ngrok 연결\n",
        "\n"
      ],
      "metadata": {
        "id": "R43PqMTEYJ8a"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from flask import Flask, request\n",
        "import requests\n",
        "app = Flask(__name__)\n",
        "FB_API_URL = \n",
        "VERIFY_TOKEN=\n",
        "PAGE_ACCESS_TOKEN=\n",
        "def send_message(recipient_id, text):\n",
        "    \"\"\"Send a response to Facebook\"\"\"\n",
        "    payload = {\n",
        "        'message': {\n",
        "            'text': text\n",
        "        },\n",
        "        'recipient': {\n",
        "            'id': recipient_id\n",
        "        },\n",
        "        'notification_type': 'regular'\n",
        "    }\n",
        "\n",
        "    auth = {\n",
        "        'access_token': PAGE_ACCESS_TOKEN\n",
        "    }\n",
        "\n",
        "    response = requests.post(\n",
        "        FB_API_URL,\n",
        "        params=auth,\n",
        "        json=payload\n",
        "    )\n",
        "\n",
        "    return response.json()\n",
        "\n",
        "def get_bot_response(message):\n",
        "    \"\"\"This is just a dummy function, returning a variation of what\n",
        "    the user said. Replace this function with one connected to chatbot.\"\"\"\n",
        "    return \"This is a dummy response to '{}'\".format(message)\n",
        "\n",
        "\n",
        "def verify_webhook(req):\n",
        "    if req.args.get(\"hub.verify_token\") == VERIFY_TOKEN:\n",
        "        return req.args.get(\"hub.challenge\")\n",
        "    else:\n",
        "        return \"incorrect\"\n",
        "\n",
        "def respond(sender, message):\n",
        "    \"\"\"Formulate a response to the user and\n",
        "    pass it on to a function that sends it.\"\"\"\n",
        "    response = get_bot_response(message)\n",
        "    send_message(sender, response)\n",
        "\n",
        "\n",
        "def is_user_message(message):\n",
        "    \"\"\"Check if the message is a message from the user\"\"\"\n",
        "    return (message.get('message') and\n",
        "            message['message'].get('text') and\n",
        "            not message['message'].get(\"is_echo\"))\n",
        "\n",
        "\n",
        "@app.route(\"/webhook\", methods=['GET'])\n",
        "def listen():\n",
        "    \"\"\"This is the main function flask uses to\n",
        "    listen at the `/webhook` endpoint\"\"\"\n",
        "    if request.method == 'GET':\n",
        "        return verify_webhook(request)\n",
        "\n",
        "@app.route(\"/webhook\", methods=['POST'])\n",
        "def talk():\n",
        "    payload = request.get_json()\n",
        "    event = payload['entry'][0]['messaging']\n",
        "    for x in event:\n",
        "        if is_user_message(x):\n",
        "            text = x['message']['text']\n",
        "            sender_id = x['sender']['id']\n",
        "            respond(sender_id, text)\n",
        "\n",
        "    return \"ok\"\n",
        "\n",
        "@app.route('/')\n",
        "def hello():\n",
        "    return 'hello'\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    app.run(threaded=True, port=5000)\n"
      ],
      "metadata": {
        "id": "Cb69wYHXZHKu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "ngrok과 페이스북 메세지 연결 코드"
      ],
      "metadata": {
        "id": "E1oM2s4HZKI4"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "- Open Api를 이용하여 정보를 수집하고 이를 Dialog Flow와 연결하여 챗봇이 사용자의 니즈에 맞는 정보를 제공한다. 본 챗봇의 경우 관광지 데이터를 공공데이터포털 에서 제공받아 이를 Dialog Flow와 연결하여 사용자에게 정보를 제공한다.\n",
        "\n",
        "  \n",
        "\n",
        "- 공공데이터포털에서 제공하지 않는 정보들이나 인터넷에 있는 정보를 활용하기 위하여 Web crawling 기술을 활용해 더 많은 정보를 제공한다.\n",
        "\n",
        "- Dialog Flow 대화문 작성"
      ],
      "metadata": {
        "id": "MQd4CyGCZPBF"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# -*- coding: utf-8 -*-\n",
        "\n",
        "\n",
        "\n",
        "# 파일로 출력하기\n",
        "\n",
        "i = 1\n",
        "\n",
        "# 출력, 입력 값 JSON 파일을 생성합니다.\n",
        "\n",
        "prev = str(conversations[0].contentName) + str(conversations[0].contentType)\n",
        "\n",
        "f = open(prev + '.json', 'w', encoding='UTF-8')\n",
        "\n",
        "f.write('{ \"id\": \"10d3155d-4468-4118-8f5d-15009af446d0\", \"name\": \"' + prev + '\", \"auto\": true, \"contexts\": [], \"responses\": [ { \"resetContexts\": false, \"affectedContexts\": [], \"parameters\": [], \"messages\": [ { \"type\": 0, \"lang\": \"en\", \"speech\": \"' + conversations[0].answer + '\" } ], \"defaultResponsePlatforms\": {}, \"speech\": [] } ], \"priority\": 500000, \"webhookUsed\": false, \"webhookForSlotFilling\": false, \"fallbackIntent\": false, \"events\": [] }')\n",
        "\n",
        "f.close()\n",
        "\n",
        "f = open(prev + '_usersays_en.json', 'w', encoding='UTF-8')\n",
        "\n",
        "f.write(\"[\")\n",
        "\n",
        "f.write('{ \"id\": \"3330d5a3-f38e-48fd-a3e6-000000000001\", \"data\": [ { \"text\": \"' + conversations[0].question + '\", \"userDefined\": false } ], \"isTemplate\": false, \"count\": 0 },')\n",
        "\n",
        "\n",
        "\n",
        "while True:\n",
        "\n",
        "    if i >= len(conversations):\n",
        "\n",
        "        f.write(\"]\")\n",
        "\n",
        "        f.close()\n",
        "\n",
        "        break;\n",
        "\n",
        "    c = conversations[i]\n",
        "\n",
        "    if prev == str(c.contentName) + str(c.contentType):\n",
        "\n",
        "        f.write('{ \"id\": \"3330d5a3-f38e-48fd-a3e6-000000000001\", \"data\": [ { \"text\": \"' + c.question + '\", \"userDefined\": false } ], \"isTemplate\": false, \"count\": 0 }')\n",
        "\n",
        "    else:\n",
        "\n",
        "        f.write(\"]\")\n",
        "\n",
        "        f.close()\n",
        "\n",
        "        # 출력, 입력 값 JSON 파일을 생성합니다.\n",
        "\n",
        "        prev = str(c.contentName) + str(c.contentType)\n",
        "\n",
        "        f = open(prev + '.json', 'w', encoding='UTF-8')\n",
        "\n",
        "        f.write('{ \"id\": \"10d3155d-4468-4118-8f5d-15009af446d0\", \"name\": \"' + prev + '\", \"auto\": true, \"contexts\": [], \"responses\": [ { \"resetContexts\": false, \"affectedContexts\": [], \"parameters\": [], \"messages\": [ { \"type\": 0, \"lang\": \"en\", \"speech\": \"' + c.answer + '\" } ], \"defaultResponsePlatforms\": {}, \"speech\": [] } ], \"priority\": 500000, \"webhookUsed\": false, \"webhookForSlotFilling\": false, \"fallbackIntent\": false, \"events\": [] }')\n",
        "\n",
        "        f.close()\n",
        "\n",
        "        f = open(prev + '_usersays_en.json', 'w', encoding='UTF-8')\n",
        "\n",
        "        f.write(\"[\")\n",
        "\n",
        "        f.write('{ \"id\": \"3330d5a3-f38e-48fd-a3e6-000000000001\", \"data\": [ { \"text\": \"' + c.question + '\", \"userDefined\": false } ], \"isTemplate\": false, \"count\": 0 }')\n",
        "\n",
        "    i = i + 1"
      ],
      "metadata": {
        "id": "7izsXZP6ZTWD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "코드 실행 결과\n",
        "![img](https://ifh.cc/g/Gsd3c1.png)"
      ],
      "metadata": {
        "id": "Qfct-1R7ZZtI"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### HW 구성도\n",
        "\n",
        "- UX/UI : Facebook Messenger\n",
        "\n",
        "![img](https://ifh.cc/g/yRXy20.png)\n",
        "\n",
        "- Database : MongoDB\n",
        "\n",
        "![img](https://ifh.cc/g/kpzoTc.png)\n",
        "\n",
        "- 챗봇은 데이터를 보관하고 분석하는 AWS 서버를 활용하여 사용자에게 데이터를 제공한다.\n",
        "\n",
        "- 또한, Amazon EC2를 이용하여 원하는 수의 가상 서버를 구축하고 보안 및 네트워킹 을 구성하며 스토리지를 관리한다.\n",
        "\n",
        "\n",
        "![img](https://ifh.cc/g/hrS7hL.png)\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "qkAAQNRTZj0T"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **서비스 흐름도**"
      ],
      "metadata": {
        "id": "zjMQkr0daOMl"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "- 페이스북이 제공하는 기능(카테고리 등)을 이용해 사용자가 이용하기 편한 화면으로 구성한다.\n",
        "- NLP Engine으로 Dialog Flov를 활용한다. nerok을 이용해 서버를 만들어 DialogFlow의 Webhook을 연결한다.\n",
        "- 이후 페이스북 메신저와 DialogFlow를 연결해\n",
        "  UX/UI를 구성하고, 백엔드 서비스를 띄워 답변 제공이 가능하게 한다.\n",
        "\n",
        "![img](https://ifh.cc/g/FnF4tb.png)"
      ],
      "metadata": {
        "id": "Hh5VwTloaUsb"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **메뉴 구성도**\n",
        "\n",
        "\n",
        "![img](https://ifh.cc/g/aw6YO0.png)"
      ],
      "metadata": {
        "id": "5ZpiGPwkaj8E"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **엔티티 관계도**\n",
        "\n",
        "![img](https://ifh.cc/g/vn4Xqc.png)"
      ],
      "metadata": {
        "id": "p7zFoNLVaxfN"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **알고리즘 시나리오**\n",
        "\n",
        "![img](https://ifh.cc/g/FHpxr6.png)\n",
        "\n",
        "ⅰ. 사용자로부터 원하는 관광지에 관련된 키워드를 입력 받습니다.\n",
        "\n",
        "ⅱ. 관련된 키워드가 Database와 일치하는 지 확인합니다.\n",
        "\n",
        "ⅲ. 일치할 경우 챗봇이 수집한 데이터를 기반으로 사용자의 요구에 알맞은 관광정보를 추천합니다.\n",
        "\n",
        "ⅳ. 일치하지 않을 경우 대화를 종료합니다."
      ],
      "metadata": {
        "id": "-U9S0sCja_Dn"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **결과물**\n",
        "\n",
        "![img](https://ifh.cc/g/NTYW8D.png)\n",
        "![img](https://ifh.cc/g/MCn3JJ.png)\n",
        "\n",
        "[한이음 결과보고서 제출용 작품시연 영상](https://www.youtube.com/watch?v=pdcLrldZBpM)\n",
        "\n"
      ],
      "metadata": {
        "id": "wX990H9BbP_d"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "![img](https://ifh.cc/g/Hcct0g.png)"
      ],
      "metadata": {
        "id": "PfXGAUNccbnM"
      }
    }
  ]
}